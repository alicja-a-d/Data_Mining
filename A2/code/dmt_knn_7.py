# =============================================================================
# VU Data Mining Techniques 2026 - Assignment 2
# Hotel Ranking Pipeline: Preprocessing + KNN + LightGBM LambdaMART
# =============================================================================

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from datetime import datetime
from scipy import stats
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
VU_ID = "ads312"
DATA_DIR    = f"/home/{VU_ID}"           
OUTPUT_DIR  = f"/home/{VU_ID}/output"
MODELS_DIR  = f"/home/{VU_ID}/models"
FIGURES_DIR = f"/home/{VU_ID}/figures"

for d in [OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

TIMESTR = time.strftime("%Y%m%d-%H%M%S")
SEED    = 42

# Relevance mapping per competition spec:
#   booking → raw score 5, click-only → 1, nothing → 0
#   mapped to LightGBM grades 0/1/2 matching label_gain=[0,1,5]
RELEVANCE_MAP = {0: 0, 1: 1, 5: 2}

# Low-cardinality IDs only — srch_destination_id is better captured via
# engineered features to avoid overfitting on rare destination IDs
CAT_FEATURES = ["site_id", "prop_country_id", "visitor_location_country_id"]


# NOTE: srch_id and prop_id are NOT in this list — they are ID columns used
# for grouping / submission, not model inputs.  They are kept in the DataFrames
# alongside KNN_FEATURES and accessed by name where needed (train_knn, make_submission).
KNN_FEATURES = [
    # hotel quality
    "prop_starrating", "prop_review_score", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price", "prop_brand_bool",
    "promotion_flag",
    # hotel quality — interaction / discrepancy signals
    "quality_score",           # prop_starrating * prop_review_score
    "ad_vs_real",              # prop_starrating - prop_review_score (puffery gap)
    "location_score_combined", # 0.7 * loc1 + 0.3 * loc2
    # historical popularity (train-derived, no leakage)
    "estimated_position",      # inverted mean display rank (non-random searches)
    # price signals
    "price_usd", "price_ratio", "price_vs_prop_mean", "price_vs_prop_median",
    "price_vs_min", "price_vs_hist", "price_pct_rank",
    "prop_std_price",          # price volatility across historical stays
    # competitor pricing signals
    "is_cheapest_any",         # cheaper than at least one competitor
    "is_expensive_any",        # more expensive than at least one competitor
    # search context
    "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
    "srch_children_count", "srch_room_count", "srch_saturday_night_bool",
    "is_last_minute", "is_long_stay", "srch_n_hotels","random_bool",
    # temporal context (seasonality)
    "month", "day_of_week","hour",
    # within-search comparisons (most useful for KNN)
    "review_vs_max", "star_vs_max", "loc1_vs_max",
    "is_cheapest_in_srch", "is_best_review", "is_best_star",
    "starrating_pct_rank", "review_pct_rank", "loc1_rank", "loc2_rank",
    # user match
    "star_vs_hist", "bool_same_country",
    # customer type
    "new_customer",            # 1 = no booking history (affects star/price vs hist)
    # missingness flags
    "has_review_score", "has_loc_score2",
    #distance features that would not distrort manhattan distance
    "has_query_affinity", "has_distance",

]


DTYPE_MAP = {
    "srch_id": "int32", "site_id": "int16",
    "prop_id": "int32", "prop_starrating": "int8",
    "prop_brand_bool": "int8", "promotion_flag": "int8",
    "srch_length_of_stay": "int16", "srch_booking_window": "int16",
    "srch_adults_count": "int8", "srch_children_count": "int8",
    "srch_room_count": "int8", "srch_saturday_night_bool": "int8",
    "random_bool": "int8", "click_bool": "int8", "booking_bool": "int8",
    "position": "int16",
}

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(train_path: str, test_path: str):
    df   = pd.read_csv(train_path, dtype=DTYPE_MAP)
    test = pd.read_csv(test_path,  dtype=DTYPE_MAP)

    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in test.select_dtypes("float64").columns:
        test[col] = test[col].astype("float32")

    print(f"  Train: {df.shape}")
    print(f"  Test:  {test.shape}")
    return df, test


# =============================================================================
# 2. IMPUTATION
# =============================================================================

def impute(data: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Imputation for KNN: every feature column must be fully finite.
    KNeighborsClassifier rejects NaN, so every missing value must be filled.
    Binary flags capture *why* a value is missing so the fill value (0)
    is distinguishable from a genuine 0.
    All medians are derived from train_df only to prevent leakage —
    val and test rows are filled using train-derived lookup tables,
    never their own group statistics.
    """
    if train_df is None:
        train_df = data

    # --- Visitor purchase history -------------------------------------------
    # null = new customer with no booking history; fill so add_features
    # arithmetic (star_vs_hist, price_vs_hist) never sees NaN
    data["new_customer"]            = data["visitor_hist_starrating"].isnull().astype("int8")
    data["visitor_hist_starrating"] = data["visitor_hist_starrating"].fillna(0)
    data["visitor_hist_adr_usd"]    = data["visitor_hist_adr_usd"].fillna(0)

    # --- Review score --------------------------------------------------------
    # null != 0 per dataset spec; flag preserves the distinction
    data["has_review_score"]  = data["prop_review_score"].notnull().astype("int8")
    data["prop_review_score"] = data["prop_review_score"].fillna(0)

    # --- Location score 2 ----------------------------------------------------
    # Median computed from train_df per destination, then mapped onto data.
    # This ensures val/test rows use train-derived medians, not their own.
    data["has_loc_score2"] = data["prop_location_score2"].notnull().astype("int8")
    dest_loc2_median = (train_df.groupby("srch_destination_id")["prop_location_score2"]
                        .median())
    global_loc2_median = train_df["prop_location_score2"].median()
    data["prop_location_score2"] = (
        data["prop_location_score2"]
        .fillna(data["srch_destination_id"].map(dest_loc2_median))
        .fillna(global_loc2_median)
    )

    # --- Query affinity ------------------------------------------------------
    # null = hotel not indexed in internet searches
    data["has_query_affinity"]        = data["srch_query_affinity_score"].notnull().astype("int8")
    data["srch_query_affinity_score"] = data["srch_query_affinity_score"].fillna(0)

    # --- Origin-destination distance -----------------------------------------
    # Same pattern: train-derived destination medians mapped onto data.
    data["has_distance"] = data["orig_destination_distance"].notnull().astype("int8")
    dest_dist_median   = (train_df.groupby("srch_destination_id")["orig_destination_distance"]
                          .median())
    global_dist_median = train_df["orig_destination_distance"].median()
    data["orig_destination_distance"] = (
        data["orig_destination_distance"]
        .fillna(data["srch_destination_id"].map(dest_dist_median))
        .fillna(global_dist_median)
    )

    # --- Historical list price -----------------------------------------------
    # Rare missingness; global median fallback
    if "prop_log_historical_price" in data.columns:
        hist_price_median = train_df["prop_log_historical_price"].median()
        data["prop_log_historical_price"] = (
            data["prop_log_historical_price"].fillna(hist_price_median)
        )

    return data


# =============================================================================
# 3. COMPETITOR AGGREGATION
# =============================================================================
## CHECK
def agg_competitors(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse 8x3 competitor columns into 4 summary signals."""
    rate_cols = [f"comp{i}_rate" for i in range(1, 9)]

    #data["comp_score_sum"]   = data[rate_cols].sum(axis=1, skipna=True)
    data["is_cheapest_any"]  = (data[rate_cols] == 1).any(axis=1).astype("int8")
    data["is_expensive_any"] = (data[rate_cols] == -1).any(axis=1).astype("int8")
    #data["comp_advantage"]   = (
    #    (data[rate_cols] == 1).sum(axis=1) - (data[rate_cols] == -1).sum(axis=1)
    #)

    drop_cols = [f"comp{i}_{s}" for i in range(1, 9)
                 for s in ["rate", "inv", "rate_percent_diff"]]
    data.drop(columns=drop_cols, inplace=True, errors="ignore")
    return data


# =============================================================================
# 4. DATETIME FEATURES
# =============================================================================

def add_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(data["date_time"])
    data["month"]       = dt.dt.month.astype("int8")
    data["day_of_week"] = dt.dt.dayofweek.astype("int8")
    data["hour"]        = dt.dt.hour.astype("int8")
    data["year"]        = dt.dt.year.astype("int8")
    return data

# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

def add_features(data: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Feature engineering.

    All arithmetic is NaN-safe: impute() must be called first so that
    visitor_hist_*, prop_review_score, prop_location_score2, and
    prop_log_historical_price are already filled.

    Computes:
    - Property price stats (all data — no labels, no leakage risk).
    - Estimated position from non-random training displays (train_df only).
    - Within-search relative features: ranks, percentile ranks,
      best-in-search comparisons.
    - User-hotel match features (0 for new customers, not NaN).
    - Quality interaction features and context flags.

    Note: booking/click rate features are intentionally excluded — they
    use outcome labels (booking_bool, click_bool) which are unavailable
    at inference time on the test set.
    """
    if train_df is None:
        train_df = data
        all_data = data
    else:
        all_data = pd.concat([train_df, data], ignore_index=True)

    # Drop stale engineered columns to avoid duplicates on re-run
    cols_to_drop = [
        "estimated_position",
        "prop_mean_price", "prop_std_price", "prop_median_price",
        "price_vs_prop_mean", "price_vs_prop_median",
        "price_mean", "price_std", "price_ratio", "loc1_rank", "loc2_rank",
        "price_pct_rank", "starrating_pct_rank", "review_pct_rank",
        "srch_n_hotels", "star_vs_hist", "price_vs_hist", "ad_vs_real",
        "quality_score", "location_score_combined", "bool_same_country",
        "is_last_minute", "is_long_stay",
        "price_vs_min", "review_vs_max", "star_vs_max", "loc1_vs_max",
        "is_cheapest_in_srch", "is_best_review", "is_best_star",
    ]
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # --- Property-level price stats (all data — no labels, safe) -------------
    price_stats = (all_data.groupby("prop_id")
                   .agg(prop_mean_price   = ("price_usd", "mean"),
                        prop_std_price    = ("price_usd", "std"),
                        prop_median_price = ("price_usd", "median"))
                   .reset_index())

    # --- Historical display position (train, non-random rows only) -----------
    # Inverted so higher = historically shown higher = stronger prior signal
    estimated_pos = (train_df[train_df["random_bool"] == 0]
                     .groupby(["srch_destination_id", "prop_id"])["position"]
                     .mean()
                     .reset_index()
                     .rename(columns={"position": "estimated_position"}))
    estimated_pos["estimated_position"] = 1.0 / estimated_pos["estimated_position"]

    # Merge lookup tables
    data = (data
            .merge(price_stats,   on="prop_id",                          how="left")
            .merge(estimated_pos, on=["srch_destination_id", "prop_id"], how="left"))

    # --- Fill NaNs from merges -----------------------------------------------
    data["prop_std_price"]     = data["prop_std_price"].fillna(0)
    data["estimated_position"] = data["estimated_position"].fillna(0)
    # prop_mean/median_price: a hotel with no price history is extremely rare;
    # fill with the global mean so downstream arithmetic stays finite
    global_mean_price   = train_df["price_usd"].mean()
    global_median_price = train_df["price_usd"].median()
    data["prop_mean_price"]   = data["prop_mean_price"].fillna(global_mean_price)
    data["prop_median_price"] = data["prop_median_price"].fillna(global_median_price)

    # --- Price deviation from property historical average --------------------
    data["price_vs_prop_mean"]   = data["price_usd"] - data["prop_mean_price"]
    data["price_vs_prop_median"] = data["price_usd"] - data["prop_median_price"]

    # --- Within-search features ----------------------------------------------
    g = data.groupby("srch_id")

    data["srch_n_hotels"] = g["prop_id"].transform("count")
    data["price_mean"]    = g["price_usd"].transform("mean")
    data["price_std"]     = g["price_usd"].transform("std").fillna(0)
    data["price_ratio"]   = data["price_usd"] / (data["price_mean"] + 1e-6)

    # Dense ranks (rank 1 = best within this search)
    data["loc1_rank"] = g["prop_location_score1"].transform(
                            "rank", ascending=False, method="dense")
    data["loc2_rank"] = g["prop_location_score2"].transform(
                            "rank", ascending=False, method="dense")

    # Percentile ranks (0–1): comparable across searches of different sizes
    data["price_pct_rank"]      = g["price_usd"].transform("rank", pct=True)
    data["starrating_pct_rank"] = g["prop_starrating"].transform(
                                      "rank", ascending=False, pct=True)
    data["review_pct_rank"]     = g["prop_review_score"].transform(
                                      "rank", ascending=False, pct=True)

    # --- User-hotel match ----------------------------------------------------
    # visitor_hist_* are already filled to 0 by impute() for new customers,
    # so arithmetic is safe without np.where; new_customer flag covers it
    data["star_vs_hist"]  = data["prop_starrating"] - data["visitor_hist_starrating"]
    data["price_vs_hist"] = data["price_usd"]       - data["visitor_hist_adr_usd"]

    # --- Quality interactions -------------------------------------------------
    # prop_review_score already filled to 0 by impute()
    data["ad_vs_real"]              = data["prop_starrating"] - data["prop_review_score"]
    data["quality_score"]           = data["prop_starrating"] * data["prop_review_score"]
    data["location_score_combined"] = (data["prop_location_score1"] * 0.7 +
                                       data["prop_location_score2"] * 0.3)

    # --- Context flags -------------------------------------------------------
    data["bool_same_country"] = (
        data["visitor_location_country_id"] == data["prop_country_id"]
    ).astype("int8")
    data["is_last_minute"] = (data["srch_booking_window"] <= 3.65).astype("int8")
    data["is_advanced_booking"] =(data["srch_booking_window"] >= 17).astype("int8")
    data["is_long_stay"]   = (data["srch_length_of_stay"] >= 7).astype("int8")

    # --- Within-search best comparisons --------------------------------------
    data["price_vs_min"]  = data["price_usd"] - g["price_usd"].transform("min")
    data["review_vs_max"] = g["prop_review_score"].transform("max") - data["prop_review_score"]
    data["star_vs_max"]   = g["prop_starrating"].transform("max")   - data["prop_starrating"]
    data["loc1_vs_max"]   = g["prop_location_score1"].transform("max") - data["prop_location_score1"]

    # Binary: is this the best option in the search?
    data["is_cheapest_in_srch"] = (
        data["price_usd"] == g["price_usd"].transform("min")
    ).astype("int8")
    data["is_best_review"] = (
        data["prop_review_score"] == g["prop_review_score"].transform("max")
    ).astype("int8")
    data["is_best_star"] = (
        data["prop_starrating"] == g["prop_starrating"].transform("max")
    ).astype("int8")

    # --- Final NaN audit (catches any remaining gaps before KNN sees data) ---
    knn_cols = [c for c in KNN_FEATURES if c in data.columns]
    nan_counts = data[knn_cols].isnull().sum()
    remaining = nan_counts[nan_counts > 0]
    if not remaining.empty:
        raise ValueError(
            f"NaNs remain in KNN_FEATURES after add_features — fix imputation:\n"
            f"{remaining.to_string()}"
        )

    return data

# =============================================================================
# 6. FULL PREPROCESSING WRAPPER
# =============================================================================

def preprocess(df: pd.DataFrame, test: pd.DataFrame):
    df.drop(columns=["gross_bookings_usd"], inplace=True, errors="ignore")

    for data in [df, test]:
        add_datetime_features(data)

    df   = impute(df,   train_df=df)
    test = impute(test, train_df=df)

    df   = agg_competitors(df)
    test = agg_competitors(test)

    return df, test


# =============================================================================
# 7. TRAIN / VALIDATION SPLIT
# =============================================================================

def split_train_val(df: pd.DataFrame, val_fraction: float = 0.2):
    """Split by srch_id so complete searches stay in one set."""
    srch_ids = df["srch_id"].unique()
    np.random.seed(SEED)
    np.random.shuffle(srch_ids)
    split     = int(len(srch_ids) * (1 - val_fraction))
    train_ids = srch_ids[:split]
    val_ids   = srch_ids[split:]

    train = df[df["srch_id"].isin(train_ids)].copy().reset_index(drop=True)
    val   = df[df["srch_id"].isin(val_ids)].copy().reset_index(drop=True)

    print(f"  Train: {len(train):,} rows | {train['srch_id'].nunique():,} searches")
    print(f"  Val:   {len(val):,} rows   | {val['srch_id'].nunique():,} searches")
    return train, val


# =============================================================================
# 8. RELEVANCE LABELS
# =============================================================================

def add_relevance(data: pd.DataFrame) -> pd.DataFrame:
    raw = data["click_bool"] + 4 * data["booking_bool"]  # 0, 1, or 5
    data["relevance"] = raw.map(RELEVANCE_MAP).astype("int8")
    return data


# =============================================================================
# 9. EVALUATION
# =============================================================================

def eval_ndcg5(df: pd.DataFrame, score_col: str = "score",
               relevance_col: str = "relevance") -> float:
    scores = []
    for _, grp in df.groupby("srch_id"):
        true_rel = grp[relevance_col].values
        pred_rel = grp[score_col].values
        if len(true_rel) > 1 and true_rel.sum() > 0:
            scores.append(ndcg_score([true_rel], [pred_rel], k=5))
    return float(np.mean(scores))

# =============================================================================
# 10. MODEL: KNN RANKING
# =============================================================================


def train_knn(train_fit: pd.DataFrame,
              val:   pd.DataFrame,
              features: list):
    """
    Train a KNN ranking model assed in as

    Parameters
    ----------
    train        : training DataFrame (already split, features engineered)
    val          : validation DataFrame
    features     : list of feature column names (no srch_id)

    Returns
    -------
    best_model  : fitted Pipeline (scaler + knn) with best hyperparameters
    ndcg_val    : NDCG@5 on the validation set
    """

    # ------------------------------------------------------------------
    # Subsample to make CV feasible (sample by srch_id to keep searches intact)
    # ------------------------------------------------------------------
    unique_ids = train_fit["srch_id"].unique()
    sample_ids = pd.Series(unique_ids).sample(
        n=min(5000, len(unique_ids)), random_state=SEED
    )
    df_sample  = train_fit[train_fit["srch_id"].isin(sample_ids)].copy()

    X_train = df_sample[features]
    y_train = df_sample["relevance"]
    groups  = df_sample["srch_id"]

    X_val = val[features]

    # ------------------------------------------------------------------
    # NDCG@5 scorer — operates at the search level, not row level
    # ------------------------------------------------------------------
    def ndcg_scorer_fn(estimator, X, y):
        proba     = estimator.predict_proba(X)
        classes   = estimator.named_steps["knn"].classes_
        score_vec = np.zeros(len(X))
        if 2 in classes:
            score_vec += proba[:, list(classes).index(2)] * 1.0
        if 1 in classes:
            score_vec += proba[:, list(classes).index(1)] * 0.1

        tmp = pd.DataFrame({
            "score":     score_vec,
            "relevance": y.values,
            "srch_id":   df_sample.loc[y.index, "srch_id"].values,
        })

        ndcg_scores = []
        for _, grp in tmp.groupby("srch_id"):
            if grp["relevance"].sum() == 0:
                continue
            ndcg_scores.append(
                ndcg_score(
                    grp["relevance"].values.reshape(1, -1),
                    grp["score"].values.reshape(1, -1),
                    k=5,
                )
            )
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    # ------------------------------------------------------------------
    # Pipeline — scaler inside pipeline prevents leakage across CV folds
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier()),
    ])

    param_grid = {
        "scaler": [StandardScaler(), RobustScaler()],
        "knn__n_neighbors": [40, 50, 65, 80, 90, 100,300],
        "knn__weights":     ["distance", "uniform"],
        "knn__metric":      ["manhattan", "minkowski"],
    }

    cv_strategy = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        pipeline,
        param_grid = param_grid,
        cv         = cv_strategy,
        scoring    = ndcg_scorer_fn,
        n_jobs     = -1,
        verbose    = 2,
        refit      = True,
    )

    print("\nStarting KNN Cross-Validation Grid Search...")
    grid_search.fit(X_train, y_train, groups=groups)

    print(f"  Best CV NDCG@5:  {grid_search.best_score_:.4f}")
    print(f"  Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # Evaluate on validation set
    # ------------------------------------------------------------------
    proba_val   = best_model.predict_proba(X_val)
    classes     = best_model.named_steps["knn"].classes_
    score_val   = np.zeros(len(X_val))
    if 2 in classes:
        score_val += proba_val[:, list(classes).index(2)] * 1.0
    if 1 in classes:
        score_val += proba_val[:, list(classes).index(1)] * 0.1

    val_tmp = val[["srch_id", "relevance"]].copy()
    val_tmp["score"] = score_val

    ndcg_scores = []
    for _, grp in val_tmp.groupby("srch_id"):
        if grp["relevance"].sum() == 0:
            continue
        ndcg_scores.append(
            ndcg_score(
                grp["relevance"].values.reshape(1, -1),
                grp["score"].values.reshape(1, -1),
                k=5
            )
        )
    ndcg_val = float(np.mean(ndcg_scores)) if ndcg_scores else float("nan")
    print(f"  Validation NDCG@5: {ndcg_val:.4f}")

    return best_model, ndcg_val


def knn_neighbour_sweep(
    train_fit: pd.DataFrame,
    val: pd.DataFrame,
    features: list,
    ks: list = None,
    output_path: str = "knn_neighbour_sweep.png",
) -> pd.DataFrame:
    """
    Sweeps over a range of k values using the best configuration found
    in train_knn (RobustScaler, manhattan, distance weighting) and plots
    CV NDCG@5 vs k to identify the elbow and justify stopping point.

    Parameters
    ----------
    train_fit   : training DataFrame
    val         : validation DataFrame
    features    : feature list
    ks          : list of k values to sweep
    output_path : path to save the plot

    Returns
    -------
    DataFrame with columns: k, cv_ndcg, val_ndcg, delta_cv
    """
    if ks is None:
        ks = list(range(100, 600, 40))

    # ------------------------------------------------------------------
    # Subsample — identical logic to train_knn for consistency
    # ------------------------------------------------------------------
    unique_ids = train_fit["srch_id"].unique()
    sample_ids = pd.Series(unique_ids).sample(
        n=min(4000, len(unique_ids)), random_state=SEED
    )
    df_sample = train_fit[train_fit["srch_id"].isin(sample_ids)].copy()

    X_train = df_sample[features]
    y_train = df_sample["relevance"]
    groups  = df_sample["srch_id"]
    X_val   = val[features]

    # ------------------------------------------------------------------
    # Search size statistics — printed for report justification
    # ------------------------------------------------------------------
    search_sizes    = train_fit.groupby("srch_id")[features[0]].count()
    avg_search_size = search_sizes.mean()
    med_search_size = search_sizes.median()
    max_search_size = search_sizes.max()

    print("\n" + "="*70)
    print("KNN NEIGHBOUR SWEEP")
    print("="*70)
    print(f"  Fixed config: RobustScaler | manhattan | distance weighting")
    print(f"\n  Search size statistics (full training set):")
    print(f"    Mean hotels per search:   {avg_search_size:.1f}")
    print(f"    Median hotels per search: {med_search_size:.1f}")
    print(f"    Max hotels per search:    {max_search_size:.1f}")
    print(f"\n  Note: KNN learns a global booking probability conditioned on")
    print(f"  hotel features. Cross-search neighbours are expected and correct")
    print(f"  — the model generalises across searches rather than memorising")
    print(f"  search-specific patterns.")
    print(f"\n  Reference k thresholds:")
    for mult in [5, 10]:
        print(f"    {mult}x mean search size: k = {int(avg_search_size * mult)}")

    # ------------------------------------------------------------------
    # NDCG@5 scorer — identical to train_knn
    # ------------------------------------------------------------------
    def ndcg_scorer_fn(estimator, X, y):
        proba     = estimator.predict_proba(X)
        classes   = estimator.named_steps["knn"].classes_
        score_vec = np.zeros(len(X))
        if 2 in classes:
            score_vec += proba[:, list(classes).index(2)] * 1.0
        if 1 in classes:
            score_vec += proba[:, list(classes).index(1)] * 0.1

        tmp = pd.DataFrame({
            "score":     score_vec,
            "relevance": y.values,
            "srch_id":   df_sample.loc[y.index, "srch_id"].values,
        })

        ndcg_scores = []
        for _, grp in tmp.groupby("srch_id"):
            if grp["relevance"].sum() == 0:
                continue
            ndcg_scores.append(
                ndcg_score(
                    grp["relevance"].values.reshape(1, -1),
                    grp["score"].values.reshape(1, -1),
                    k=5,
                )
            )
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    cv_strategy = GroupKFold(n_splits=5)

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    print(f"\n  Sweeping k = {ks}")
    print(f"\n  {'k':<8} {'CV NDCG@5':>12} {'Val NDCG@5':>12} "
          f"{'Δ CV':>10} {'Rolling Δ':>12}")
    print(f"  {'-'*58}")

    rows         = []
    prev_cv_ndcg = None

    for k in ks:
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("knn",    KNeighborsClassifier(
                           n_neighbors = k,
                           weights     = "distance",
                           metric      = "manhattan",
            )),
        ])

        # CV NDCG
        cv_scores = cross_val_score(
            pipe,
            X_train, y_train,
            groups  = groups,
            cv      = cv_strategy,
            scoring = ndcg_scorer_fn,
            n_jobs  = -1,
        )
        cv_ndcg = cv_scores.mean()

        # Validation NDCG
        pipe.fit(X_train, y_train)

        proba_val = pipe.predict_proba(X_val)
        classes   = pipe.named_steps["knn"].classes_
        score_val = np.zeros(len(X_val))
        if 2 in classes:
            score_val += proba_val[:, list(classes).index(2)] * 1.0
        if 1 in classes:
            score_val += proba_val[:, list(classes).index(1)] * 0.1

        val_tmp          = val[["srch_id", "relevance"]].copy()
        val_tmp["score"] = score_val

        val_ndcg_scores = []
        for _, grp in val_tmp.groupby("srch_id"):
            if grp["relevance"].sum() == 0:
                continue
            val_ndcg_scores.append(
                ndcg_score(
                    grp["relevance"].values.reshape(1, -1),
                    grp["score"].values.reshape(1, -1),
                    k=5,
                )
            )
        val_ndcg = float(np.mean(val_ndcg_scores)) if val_ndcg_scores else float("nan")

        delta = (cv_ndcg - prev_cv_ndcg) if prev_cv_ndcg is not None else 0.0

        rows.append({
            "k":        k,
            "cv_ndcg":  cv_ndcg,
            "val_ndcg": val_ndcg,
            "delta_cv": delta,
        })
        prev_cv_ndcg = cv_ndcg

        # Rolling mean printed live — computed from accumulated rows
        if len(rows) >= 3:
            rolling_delta = np.mean([r["delta_cv"] for r in rows[-3:]])
        else:
            rolling_delta = delta

        print(f"  {k:<8} {cv_ndcg:>12.4f} {val_ndcg:>12.4f} "
              f"{delta:>+10.4f} {rolling_delta:>+12.4f}")
        

    results_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Elbow detection — rolling mean of delta over 3 steps
    # Skip first row (delta always 0) and find where rolling gain
    # falls and stays below threshold
    # ------------------------------------------------------------------
    ndcg_threshold = 0.001

    results_df["delta_cv_rolling"] = (
        results_df["cv_ndcg"]
        .diff()
        .rolling(3, min_periods=2)
        .mean()
        .abs()
    )

    # Only look from the 3rd row onwards to avoid false early trigger
    candidates = results_df.iloc[2:][
        results_df.iloc[2:]["delta_cv_rolling"] < ndcg_threshold
    ]

    if not candidates.empty:
        elbow_k = int(candidates.iloc[0]["k"])
        elbow_source = f"rolling Δ < {ndcg_threshold}"
    else:
        # No elbow found — use last point and flag it
        elbow_k = ks[-1]
        elbow_source = "no convergence detected — extend sweep"

    print(f"\n  Elbow ({elbow_source}): k = {elbow_k}")
    print(f"  Best CV NDCG@5:  {results_df['cv_ndcg'].max():.4f} "
          f"at k={results_df.loc[results_df['cv_ndcg'].idxmax(), 'k']}")
    print(f"  Best Val NDCG@5: {results_df['val_ndcg'].max():.4f} "
          f"at k={results_df.loc[results_df['val_ndcg'].idxmax(), 'k']}")

    if elbow_k == ks[-1]:
        print(f"\n  [Warning] NDCG has not converged within k={ks[-1]}.")
        print(f"  Consider extending the sweep to k={ks[-1] + 100}.")

    # ------------------------------------------------------------------
    # Plot — single panel, clean
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # NDCG lines
    ax1.plot(results_df["k"], results_df["cv_ndcg"],
             marker="o", linewidth=2, markersize=5,
             label="CV NDCG@5", color="steelblue")
    ax1.plot(results_df["k"], results_df["val_ndcg"],
             marker="s", linewidth=2, markersize=5,
             linestyle="--", label="Validation NDCG@5",
             color="coral")

    # Elbow line — only if genuinely detected
    if elbow_k != ks[-1]:
        ax1.axvline(x=elbow_k, color="gray", linestyle=":",
                    linewidth=1.5,
                    label=f"Elbow — rolling Δ < {ndcg_threshold} (k={elbow_k})")

    # Reference lines — mean search size multiples
    ax1.axvline(x=int(avg_search_size * 5), color="green",
                linestyle="--", linewidth=1, alpha=0.7,
                label=f"5× mean search size (k={int(avg_search_size*5)})")
    ax1.axvline(x=int(avg_search_size * 10), color="orange",
                linestyle="--", linewidth=1, alpha=0.7,
                label=f"10× mean search size (k={int(avg_search_size*10)})")

    ax1.set_xlabel("Number of neighbours (k)", fontsize=12)
    ax1.set_ylabel("NDCG@5", fontsize=12)
    ax1.set_title("KNN neighbour sweep — NDCG@5 vs k\n"
                  "RobustScaler | Manhattan distance | Distance weighting",
                  fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Marginal gain as bars on secondary axis
    ax2 = ax1.twinx()
    bar_width = (ks[1] - ks[0]) * 0.6 if len(ks) > 1 else 10
    ax2.bar(results_df["k"].iloc[1:],
            results_df["delta_cv"].iloc[1:].clip(lower=0),
            alpha=0.15, color="steelblue",
            width=bar_width, label="Marginal CV gain (Δ per step)")
    ax2.set_ylabel("Marginal NDCG gain (Δ per step)",
                   fontsize=10, color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(bottom=0)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to: {output_path}")

    return results_df

# =============================================================================
# 11. RETRAIN KNN ON FULL DATA
# =============================================================================

def retrain_knn_on_full_data(df_full: pd.DataFrame,
                              best_params: dict,
                              features: list) -> object:
    """
    Retrain the KNN pipeline on 100% of labelled data using the best
    hyper-parameters found during cross-validation.

    Parameters
    ----------
    df_full      : full labelled DataFrame (train + val, features already added)
    best_params  : dict from GridSearchCV.best_params_
    features     : list of feature column names (KNN_FEATURES)

    Returns
    -------
    Fitted Pipeline (scaler + knn)
    """
    print(f"\n--- Retraining KNN on full data ({len(df_full):,} rows, "
          f"{len(features)} features, ")
    print(f"  Best params: {best_params}")

    missing = [f for f in features if f not in df_full.columns]
    if missing:
        raise ValueError(f"Missing features in full dataset: {missing}")

    df_s = df_full.sort_values("srch_id").reset_index(drop=True)

    X = df_s[features]
    y = df_s["relevance"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(
            n_neighbors = best_params.get("knn__n_neighbors", 20),
            weights     = best_params.get("knn__weights",     "distance"),
            metric      = best_params.get("knn__metric",      "manhattan"),
        )),
    ])

    pipeline.fit(X, y)
    print("  Done.")
    return pipeline


# =============================================================================
# 12. SUBMISSION GENERATION
# =============================================================================

def make_submission(test: pd.DataFrame, model,
                    model_name: str = "knn",
                    features: list = None) -> pd.DataFrame:
    """
    Generate a submission CSV from a fitted KNN pipeline.
    Scores each row as: P(booked) + 0.1 * P(clicked), then ranks
    within each search by descending score.
    """
    if features is None:
        features = KNN_FEATURES

    # Ensure srch_id and prop_id are present for grouping / output
    required_id_cols = ["srch_id", "prop_id"]
    missing_id = [c for c in required_id_cols if c not in test.columns]
    if missing_id:
        raise ValueError(f"Test DataFrame is missing ID columns: {missing_id}")

    test_s = test.sort_values("srch_id").reset_index(drop=True)

    # KNN: use probability of booking (class 2) + 0.1 * prob of click (class 1)
    # as a continuous ranking score — same logic as in train_knn evaluation
    proba   = model.predict_proba(test_s[features])
    classes = list(model.named_steps["knn"].classes_)
    score   = np.zeros(len(test_s))
    if 2 in classes:
        score += proba[:, classes.index(2)] * 1.0
    if 1 in classes:
        score += proba[:, classes.index(1)] * 0.1
    test_s["score"] = score

    submission = (test_s
                  .sort_values(["srch_id", "score"], ascending=[True, False])
                  [["srch_id", "prop_id"]])

    path = f"{OUTPUT_DIR}/VU-DM-2026-Group-62-{model_name}-{TIMESTR}.csv"
    submission.to_csv(path, index=False)
    print(f"  Submission saved: {path} ({len(submission):,} rows)")
    return submission


# =============================================================================
# 13. FAIRNESS ANALYSIS
# =============================================================================

def _score_model(model, df: pd.DataFrame, features: list, alpha: float = 0.0, target_col: str = None) -> np.ndarray:
    """
    Return a continuous ranking score for every row in df.
    Score = P(booked | class 2) + 0.1 * P(clicked | class 1) + alpha (applied to target_col == 0).
    
    Parameters
    ----------
    alpha      : float, default 0.0
                 The score bonus added to unprivileged group elements (value == 0).
    target_col : str, optional
                 The column name being evaluated for mitigation. If None, no mitigation 
                 is applied.
    """
    proba   = model.predict_proba(df[features])
    classes = list(model.named_steps["knn"].classes_)
    score   = np.zeros(len(df))
    
    if 2 in classes:
        score += proba[:, classes.index(2)] * 1.0
    if 1 in classes:
        score += proba[:, classes.index(1)] * 0.1
        
    # Apply post-processing mitigation to target column if parameter is provided
    if alpha != 0.0 and target_col is not None:
        if target_col in df.columns:
            unprivileged_mask = (df[target_col].values == 0)
            score[unprivileged_mask] += alpha
        else:
            print(f"  [Warning] Mitigation column '{target_col}' not found in target DataFrame.")
            
    return score


def compute_ndcg_by_group(val: pd.DataFrame,
                          model,
                          features: list,
                          group_col: str,
                          alpha: float = 0.0) -> pd.DataFrame:
    """
    Unified NDCG@5 computation with group labelling. Accepts an optional alpha parameter 
    to evaluate post-processing adjustments.
    """
    tmp = val[["srch_id", "relevance", group_col]].copy()
    tmp["score"] = _score_model(model, val, features, alpha=alpha, target_col = group_col)

    group_label = (val.drop_duplicates("srch_id")[["srch_id", group_col]])

    rows = []
    for srch_id, grp in tmp.groupby("srch_id"):
        if grp["relevance"].sum() == 0:
            continue  
        score = ndcg_score(
            grp["relevance"].values.reshape(1, -1),
            grp["score"].values.reshape(1, -1),
            k=5,
        )
        rows.append({"srch_id": srch_id, "ndcg5": score})

    return pd.DataFrame(rows).merge(group_label, on="srch_id", how="left")

def compute_ndcg_by_group(val: pd.DataFrame,
                          model,
                          features: list,
                          group_col: str,
                          alpha: float = 0.0) -> pd.DataFrame:
    tmp = val[["srch_id", "relevance", group_col]].copy()
    tmp["score"] = _score_model(model, val, features, alpha=alpha, target_col=None)

    rows = []
    for srch_id, grp in tmp.groupby("srch_id"):
        if grp["relevance"].sum() == 0:
            continue  # genuinely undefined — skip

        score = ndcg_score(
            grp["relevance"].values.reshape(1, -1),
            grp["score"].values.reshape(1, -1),
            k=5,
        )

        booked = grp[grp["relevance"] == 2]
        if not booked.empty:
            group_val = booked[group_col].iloc[0]
        elif grp[group_col].nunique() == 1:
            # no booking but clicked — safe for search-level attributes
            group_val = grp[group_col].iloc[0]
        else:
            # clicked-only search with mixed hotel-level attribute values
            # — ambiguous which group to assign, skip
            continue

        rows.append({"srch_id": srch_id, "ndcg5": score, group_col: group_val})

    return pd.DataFrame(rows)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size: (mean_a - mean_b) / pooled_std."""
    n_a, n_b   = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.std(ddof=1)**2 +
                          (n_b - 1) * b.std(ddof=1)**2) /
                         (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def _bootstrap_ci(a: np.ndarray, b: np.ndarray,
                  n_boot: int = 2000,
                  ci: float = 95.0,
                  seed: int = SEED) -> tuple:
    """Bootstrap 95% CI on the difference in means (mean_a - mean_b)."""
    rng  = np.random.default_rng(seed)
    diffs = np.array([
        rng.choice(a, size=len(a), replace=True).mean() -
        rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha_pct = (100 - ci) / 2
    return float(np.percentile(diffs, alpha_pct)), float(np.percentile(diffs, 100 - alpha_pct))


def bias_report(result: pd.DataFrame,
                group_col: str,
                group_labels: dict,
                stage: str = "",
                n_boot: int = 2000) -> tuple:
    """Generic bias report: compares NDCG@5 between two groups."""
    vals   = list(group_labels.keys())
    names  = list(group_labels.values())
    grp_a  = result[result[group_col] == vals[0]]["ndcg5"].dropna().values
    grp_b  = result[result[group_col] == vals[1]]["ndcg5"].dropna().values

    mean_a   = grp_a.mean()
    mean_b   = grp_b.mean()
    mean_gap = mean_a - mean_b
    d        = _cohens_d(grp_a, grp_b)
    ci_lo, ci_hi = _bootstrap_ci(grp_a, grp_b, n_boot=n_boot)

    effect_label = ("negligible" if abs(d) < 0.2 else
                    "small"      if abs(d) < 0.5 else
                    "medium"     if abs(d) < 0.8 else "large")

    label = f"  [{stage}]" if stage else ""
    print(f"\n--- Bias Report {label} ---")
    print(f"  {names[0]:<30} mean NDCG@5: {mean_a:.4f}  (n={len(grp_a):,})")
    print(f"  {names[1]:<30} mean NDCG@5: {mean_b:.4f}  (n={len(grp_b):,})")
    print(f"  Mean gap ({names[0]} − {names[1]}): {mean_gap:+.4f}")
    print(f"  Cohen's d:  {d:+.4f}  [{effect_label} effect]")
    print(f"  95% bootstrap CI on gap: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    if ci_lo > 0:
        ci_verdict = f"{names[0]} consistently higher (CI entirely positive)"
    elif ci_hi < 0:
        ci_verdict = f"{names[1]} consistently higher (CI entirely negative)"
    else:
        ci_verdict = "CI spans zero — gap not reliably different from 0"
    print(f"  CI verdict: {ci_verdict}")

    if abs(d) >= 0.2 and ci_lo * ci_hi > 0:
        verdict = f"bias detected: reliable and meaningful — model favours {names[0] if mean_gap > 0 else names[1]}"
    elif ci_lo * ci_hi > 0:
        verdict = f"bias detected: reliable but small effect — model favours {names[0] if mean_gap > 0 else names[1]}"
    elif abs(d) >= 0.2:
        verdict = "suggestive: meaningful effect size but CI spans zero — uncertain"
    else:
        verdict = "no meaningful bias detected"
    print(f"  Verdict: {verdict}")

    return mean_gap, d, ci_lo, ci_hi


def compute_rank_bias(val: pd.DataFrame,
                      model,
                      features: list,
                      alpha: float = 0.0,
                      target_col: str = None) -> pd.DataFrame:
    """
    Positional bias check: evaluates mean rank gaps within searches.
    Accepts optional alpha parameter to compute adjusted bias metrics directly.
    """
    if target_col is None:
        raise ValueError("target_col must be explicitly specified (e.g., target_col='prop_brand_bool')")

    if target_col not in val.columns:
        print(f"  [Warning] Column '{target_col}' not found in validation DataFrame.")
        return pd.DataFrame(columns=["srch_id", "mean_rank_class_1", "mean_rank_class_0", "rank_gap"])

    # Isolate relevant schema and score using the updated variable-aware engine
    tmp = val[["srch_id", target_col]].copy()
    tmp["score"] = _score_model(model, val, features, alpha=alpha, target_col=target_col)
    tmp["rank"]  = tmp.groupby("srch_id")["score"].rank(ascending=False, method="average")

    rows = []
    for srch_id, grp in tmp.groupby("srch_id"):
        class_1 = grp[grp[target_col] == 1]["rank"]
        class_0 = grp[grp[target_col] == 0]["rank"]
        
        # Enforce intra-search variance constraint
        if len(class_1) == 0 or len(class_0) == 0:
            continue
            
        rows.append({
            "srch_id":             srch_id,
            "mean_rank_class_1":   class_1.mean(),
            "mean_rank_class_0":   class_0.mean(),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["rank_gap"] = result["mean_rank_class_1"] - result["mean_rank_class_0"]
    else:
        print(f"  [Warning] Rank bias not calculated for '{target_col}': all properties within every individual search share the same feature value.")
        result = pd.DataFrame(columns=["srch_id", "mean_rank_class_1", "mean_rank_class_0", "rank_gap"])
        
    return result


def rank_bias_report(result: pd.DataFrame, stage: str = "",
                     n_boot: int = 2000,
                     target_col: str = None) -> tuple:
    """Positional rank bias report using Cohen's d and bootstrap CI."""
    if target_col is None:
        raise ValueError("target_col must be explicitly specified for reporting (e.g., target_col='prop_brand_bool').")
        
    if result.empty or "rank_gap" not in result.columns or result["rank_gap"].dropna().size == 0:
        print(f"\n--- Rank Bias Report [{stage}] ---")
        print(f"  Skipped: No valid intra-search variance found for feature '{target_col}'.")
        return 0.0, 0.0, 0.0, 0.0

    gaps = result["rank_gap"].dropna().values
    mean_1 = result["mean_rank_class_1"].mean()
    mean_0 = result["mean_rank_class_0"].mean()
    mean_gap = gaps.mean()

    # Calculate standardized effect size between the two position distributions
    d = _cohens_d(
        result["mean_rank_class_1"].values,
        result["mean_rank_class_0"].values,
    )

    # One-sample bootstrap on the matched search gaps
    rng  = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(gaps, size=len(gaps), replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    effect_label = ("negligible" if abs(d) < 0.2 else
                    "small"      if abs(d) < 0.5 else
                    "medium"     if abs(d) < 0.8 else "large")

    label = f"  [{stage}]" if stage else ""
    print(f"\n--- Rank Bias Report {label} | Target: {target_col} ---")
    print(f"  Mean rank {target_col}=1 (within search): {mean_1:.3f}")
    print(f"  Mean rank {target_col}=0 (within search): {mean_0:.3f}")
    print(f"  Mean rank gap (Class 1 − Class 0): {mean_gap:+.3f}")
    print(f"  (negative = feature value 1 is ranked closer to the top)")
    print(f"  Searches with both attribute types: {len(result):,}")
    print(f"  Cohen's d (vs zero):  {d:+.4f}  [{effect_label} effect]")
    print(f"  95% bootstrap CI on mean gap: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    ci_excludes_zero  = (ci_lo * ci_hi > 0)
    meaningful_effect = (abs(d) >= 0.2)

    if ci_excludes_zero and meaningful_effect:
        verdict = f"bias detected: reliable and meaningful — {target_col}=1 ranked higher" if mean_gap < 0 else f"bias detected: reliable and meaningful — {target_col}=0 ranked higher"
    elif ci_excludes_zero:
        verdict = f"bias detected: reliable but small effect — {target_col}=1 ranked higher" if mean_gap < 0 else f"bias detected: reliable but small effect — {target_col}=0 ranked higher"
    elif meaningful_effect:
        verdict = "suggestive: meaningful effect size but CI spans zero — uncertain"
    else:
        verdict = f"no systematic positional preference detected for {target_col}"
    print(f"  Verdict: {verdict}")

    return mean_gap, d, ci_lo, ci_hi
# =============================================================================
# POST-PROCESSING MITIGATION — CALIBRATION
# =============================================================================

def calibrate_alpha(model, val: pd.DataFrame,
                    features: list,
                    alpha_grid: list = None,
                    ndcg_baseline: float = None) -> tuple:
    """
    Grid search over alpha values to find the score bonus that best closes
    the positional rank gap while keeping NDCG@5 stable.
    """
    MAX_NDCG_DROP = 0.005

    if alpha_grid is None:
        alpha_grid = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    if ndcg_baseline is None:
        ndcg_baseline = compute_ndcg_by_group(val, model, features, group_col="prop_brand_bool", alpha=0.0)["ndcg5"].mean()

    rows = []
    print(f"\n  {'Alpha':>10} | {'Rank gap':>10} | {'NDCG@5':>8} | {'ΔNDCG':>8}")
    print(f"  {'-'*50}")
    print(f"  {'0.000 (base)':>10} | {'—':>10} | {ndcg_baseline:>8.4f} | {'0.0000':>8}  ← baseline")
    
    for alpha in alpha_grid:
        gap_df = compute_rank_bias(val, model, features, alpha=alpha)
        gap = gap_df["rank_gap"].mean() if not gap_df.empty else 0.0
        
        ndcg_df = compute_ndcg_by_group(val, model, features, group_col="prop_brand_bool", alpha=alpha)
        ndcg = ndcg_df["ndcg5"].mean()
        
        rows.append({"alpha": alpha, "rank_gap": gap, "ndcg5": ndcg})
        print(f"  {alpha:>10.5f} | {gap:>+10.4f} | {ndcg:>8.4f} | {ndcg - ndcg_baseline:>+8.4f}")

    results = pd.DataFrame(rows)

    valid = results[results["ndcg5"] >= ndcg_baseline - MAX_NDCG_DROP]
    if valid.empty:
        valid = results

    best_idx   = valid["rank_gap"].abs().idxmin()
    best_alpha = float(valid.loc[best_idx, "alpha"])

    print(f"\n  Best alpha: {best_alpha:.5f}  "
          f"(rank gap: {valid.loc[best_idx, 'rank_gap']:+.4f}, "
          f"NDCG@5: {valid.loc[best_idx, 'ndcg5']:.4f})")
    return best_alpha, results



def promotion_regression_analysis(
    val: pd.DataFrame,
    control_features: list,
    target_col: str = "promotion_flag",
    outcome_col: str = "booking_bool",
    group_col: str = "srch_id",
) -> pd.DataFrame:
    """
    Conditional logistic regression to estimate the effect of target_col
    on booking probability, conditioning on search ID.

    Conditioning on srch_id eliminates all search-level confounding
    by design — only within-search variation identifies coefficients.
    This is the correct model for hotel ranking data where at most one
    hotel is booked per search.

    Only hotel-level features are identified — search-level features
    are absorbed by the conditioning and must be excluded.

    Parameters
    ----------
    val              : validation DataFrame (hotel-level rows)
    control_features : hotel-level controls only
                       (search-level features will be detected and dropped)
    target_col       : binary feature being evaluated
    outcome_col      : binary outcome (booking_bool)
    group_col        : search identifier column (srch_id)
    """

    # ------------------------------------------------------------------
    # Compute price_rel if missing
    # ------------------------------------------------------------------
    if "price_usd" in val.columns and "price_rel" not in val.columns:
        val = val.copy()
        val["price_rel"] = val.groupby(group_col)["price_usd"].transform(
            lambda x: x.rank(pct=True)
        )

    # ------------------------------------------------------------------
    # Log-transform position (cascade click model predicts log-linear
    # relationship between position and examination probability)
    # ------------------------------------------------------------------
    if "position" in val.columns and "log_position" not in val.columns:
        val = val.copy()
        val["log_position"] = np.log(val["position"].clip(lower=1))

    # Replace position with log_position in control_features if present
    control_features = [
        "log_position" if f == "position" else f
        for f in control_features
    ]

    # ------------------------------------------------------------------
    # Detect and drop search-level features
    # A feature is search-level if it has no within-search variance
    # in >95% of searches — conditional logit absorbs it and it
    # contributes no identification
    # ------------------------------------------------------------------
    hotel_level_controls = []
    dropped_features     = []

    for feat in control_features:
        if feat not in val.columns:
            dropped_features.append((feat, "not in dataframe"))
            continue
        within_var = val.groupby(group_col)[feat].nunique()
        pct_no_var = (within_var == 1).mean()
        if pct_no_var > 0.95:
            dropped_features.append((feat, f"search-level ({pct_no_var*100:.1f}% searches have no variance)"))
        else:
            hotel_level_controls.append(feat)

    all_features = [target_col] + hotel_level_controls

    # ------------------------------------------------------------------
    # Build analysis dataset
    # ------------------------------------------------------------------
    cols_needed = [group_col] + all_features + [outcome_col]
    df = val[[c for c in cols_needed if c in val.columns]].dropna()

    # Conditional logit requires within-search outcome variance
    # Drop searches where nobody booked — they contribute no information
    df = df.groupby(group_col).filter(
        lambda x: 0 < x[outcome_col].sum() < len(x)
    )

    print(f"\n{'='*65}")
    print(f"  CONDITIONAL LOGISTIC REGRESSION — {target_col.upper()}")
    print(f"{'='*65}")
    print(f"  Outcome:          {outcome_col}")
    print(f"  Target variable:  {target_col}")
    print(f"  Controls used:    {hotel_level_controls}")
    if dropped_features:
        print(f"  Features dropped:")
        for feat, reason in dropped_features:
            print(f"    - {feat}: {reason}")
    print(f"  N observations:   {len(df):,}")
    print(f"  N searches:       {df[group_col].nunique():,}")
    print(f"  Booking rate:     {df[outcome_col].mean():.4f}")
    print(f"  Model:            Conditional logistic (conditioned on {group_col})")

    # ------------------------------------------------------------------
    # Within-search variance check for all retained controls
    # ------------------------------------------------------------------
    print(f"\n  Within-search variance check:")
    for feat in all_features:
        if feat not in df.columns:
            continue
        pct = (df.groupby(group_col)[feat].nunique() == 1).mean() * 100
        print(f"    {feat:<35} {100-pct:.1f}% of searches have variance")

    # ------------------------------------------------------------------
    # Model 1 — unadjusted (target only, conditioned on search)
    # ------------------------------------------------------------------
    model_raw = ConditionalLogit(
        df[outcome_col],
        df[[target_col]],
        groups=df[group_col]
    ).fit(disp=0, method="bfgs", maxiter=200)

    # ------------------------------------------------------------------
    # Model 2 — adjusted (target + hotel-level controls)
    # ------------------------------------------------------------------
    model_adj = ConditionalLogit(
        df[outcome_col],
        df[all_features],
        groups=df[group_col]
    ).fit(disp=0, method="bfgs", maxiter=200)

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------
    def extract_results(model, label):
        params = model.params
        ci     = model.conf_int()
        pvals  = model.pvalues
        rows   = []
        for feat in params.index:
            coef  = params[feat]
            or_   = np.exp(coef)
            or_lo = np.exp(ci.loc[feat, 0])
            or_hi = np.exp(ci.loc[feat, 1])
            pval  = pvals[feat]
            sig   = ("***" if pval < 0.001 else
                     "**"  if pval < 0.01  else
                     "*"   if pval < 0.05  else "")
            rows.append({
                "model":   label,
                "feature": feat,
                "coef":    coef,
                "OR":      or_,
                "OR_lo":   or_lo,
                "OR_hi":   or_hi,
                "p_value": pval,
                "sig":     sig,
            })
        return rows

    rows = (extract_results(model_raw, "Unadjusted") +
            extract_results(model_adj, "Adjusted"))
    results_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print(f"\n  {'Feature':<30} {'Model':<14} {'Odds Ratio':>12} "
          f"{'95% CI':>22} {'p-value':>10} {'':>4}")
    print(f"  {'-'*92}")
    for _, row in results_df.iterrows():
        ci_str = f"[{row['OR_lo']:.3f}, {row['OR_hi']:.3f}]"
        print(f"  {row['feature']:<30} {row['model']:<14} {row['OR']:>12.3f} "
              f"{ci_str:>22} {row['p_value']:>10.4f} {row['sig']:>4}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    promo_raw = results_df[
        (results_df["feature"] == target_col) &
        (results_df["model"]   == "Unadjusted")
    ].iloc[0]

    promo_adj = results_df[
        (results_df["feature"] == target_col) &
        (results_df["model"]   == "Adjusted")
    ].iloc[0]

    raw_pct     = (promo_raw["OR"] - 1) * 100
    adj_pct     = (promo_adj["OR"] - 1) * 100
    attenuation = (1 - promo_adj["OR"] / promo_raw["OR"]) * 100

    print(f"\n  --- Interpretation ---")
    print(f"  Unadjusted: {target_col} associated with {raw_pct:+.1f}% higher "
          f"booking odds within search  (OR={promo_raw['OR']:.3f}, "
          f"p={promo_raw['p_value']:.4f})")
    print(f"  Adjusted:   {target_col} associated with {adj_pct:+.1f}% higher "
          f"booking odds within search, controlling for hotel-level "
          f"characteristics  (OR={promo_adj['OR']:.3f}, "
          f"p={promo_adj['p_value']:.4f})")
    print(f"  Attenuation: {attenuation:.1f}% of the within-search {target_col} "
          f"effect explained by controls")

    if promo_adj["p_value"] < 0.05:
        direction = "higher" if adj_pct > 0 else "lower"
        print(f"  Verdict: statistically significant independent within-search "
              f"effect — {adj_pct:+.1f}% {direction} booking odds after controls")
    else:
        print(f"  Verdict: no statistically significant within-search effect "
              f"after controlling for hotel-level characteristics")

     # ------------------------------------------------------------------
    # Likelihood ratio test (correct fit statistic for conditional logit)
    # ------------------------------------------------------------------
    lr_stat = -2 * (model_raw.llf - model_adj.llf)
    lr_df   = len(model_adj.params) - len(model_raw.params)
    lr_pval = stats.chi2.sf(lr_stat, df=lr_df)

    print(f"\n  --- Model fit ---")
    print(f"  Unadjusted  log-likelihood: {model_raw.llf:.2f}")
    print(f"  Adjusted    log-likelihood: {model_adj.llf:.2f}")
    print(f"  Likelihood ratio test: LR={lr_stat:.2f}, df={lr_df}, "
          f"p={lr_pval:.4e}")
    print(f"  {'Controls significantly improve model fit' if lr_pval < 0.05 else 'Controls do not significantly improve fit'}")
    print(f"  Note: AIC and pseudo-R² not reported — not appropriate "
          f"for conditional logistic regression")

    return results_df, model_raw, model_adj
# =============================================================================
# 15. MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_path = os.path.join(DATA_DIR, "training_set_VU_DM.csv")
    test_path  = os.path.join(DATA_DIR, "test_set_VU_DM.csv")
    df, test   = load_data(train_path, test_path)

    # -------------------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------------------
    print("\nPreprocessing...")
    df, test = preprocess(df, test)

    # -------------------------------------------------------------------------
    # Train / val split BEFORE feature engineering (prevent leakage)
    # -------------------------------------------------------------------------
    print("\nSplitting train/val...")
    train, val = split_train_val(df, val_fraction=0.2)

    # -------------------------------------------------------------------------
    # Feature engineering (val and test use train-only stats)
    # -------------------------------------------------------------------------
    print("\nEngineering features...")
    train = add_features(train, train_df=train)
    val   = add_features(val,   train_df=train)
    test  = add_features(test,  train_df=train)

    print(f"  Columns — train: {train.shape[1]} | val: {val.shape[1]} "
          f"| test: {test.shape[1]}")

    # Validate that all KNN_FEATURES are present after engineering
    missing_feats = [f for f in KNN_FEATURES if f not in train.columns]
    if missing_feats:
        raise ValueError(f"KNN_FEATURES missing after engineering: {missing_feats}")

    # -------------------------------------------------------------------------
    # Relevance labels
    # -------------------------------------------------------------------------
    train = add_relevance(train)
    val   = add_relevance(val)



    # =========================================================================
    # BRAND BIAS DATA ANALYSIS (Steps 1–4)
    # Run on the training set before model training to establish the data-level
    # evidence for the bias. These numbers go directly in the report.
    # =========================================================================
   
    bias_features = [
        "prop_brand_bool", "new_customer", "is_cheapest_in_srch", "is_best_review",
        "is_best_star", "random_bool", "srch_saturday_night_bool", "is_last_minute", "is_long_stay"
    ]

    print("\n" + "="*80)
    print("PART 1: DATA-LEVEL BIAS ANALYSIS (BOOKING & IMPRESSION RATES)")
    print("="*80)

    for feat in bias_features:
        if feat not in train.columns:
            continue
            
        print(f"\n--- Feature: {feat} ---")
        
        # 1. Representation & Booking/Click Rates (Steps 1, 2 & 4)
        counts = train.groupby(feat)["prop_id"].count()
        shares = counts / counts.sum()
        
        rates = train.groupby(feat)[["booking_bool", "click_bool"]].mean()
        rates.insert(0, "Impression Share", shares)
        rates = rates.rename(columns={"booking_bool": "Booking Rate", "click_bool": "Click Rate"})
        
        print(rates.to_string(float_format=lambda x: f"{x:.4f}"))
        
        if len(rates) == 2:
            b_ratio = rates.iloc[1]["Booking Rate"] / (rates.iloc[0]["Booking Rate"] + 1e-9)
            c_ratio = rates.iloc[1]["Click Rate"] / (rates.iloc[0]["Click Rate"] + 1e-9)
            print(f"  Ratio (Class 1 / Class 0) -> Booking: {b_ratio:.2f}x | Click: {c_ratio:.2f}x")

    '''
    print("\n" + "="*80)
    print("PART 2: MODEL-LEVEL BIAS ANALYSIS (RANKING & METRIC CHECKS)")
    print("="*80)

    print("\n=== Training KNN baseline (detection only) ===")
    knn_base, ndcg_base = train_knn(train, val, features=KNN_FEATURES)
    print(f"Overall validation NDCG@5: {ndcg_base:.4f}\n")

    # Run after train_knn — uses best config found there
    print("\n=== KNN Neighbour Sweep ===")
    sweep_results = knn_neighbour_sweep(
        train_fit   = train,
        val         = val,
        features    = KNN_FEATURES,
        ks          = list(range(100, 600, 20)),
        output_path = "knn_neighbour_sweep.png",
    )

    # Use elbow k to retrain final model if it differs from grid search best
    best_k_sweep = sweep_results.loc[
        sweep_results["cv_ndcg"].idxmax(), "k"
    ]
    print(f"\n  Best k from sweep: {best_k_sweep}")
    print(f"  Best CV NDCG from sweep: "
      f"{sweep_results['cv_ndcg'].max():.4f}")'''

    # =========================================================================
    # PROMOTION FLAG BIAS ANALYSIS — FINAL STRUCTURE
    # =========================================================================

    
# ==========================================================================
# FULL PROMOTION BIAS ANALYSIS — MAIN BLOCK
# ==========================================================================

    if "promotion_flag" in val.columns:

        # ------------------------------------------------------------------
        # Compute transformations once
        # ------------------------------------------------------------------
        if "price_usd" in val.columns and "price_rel" not in val.columns:
            val["price_rel"] = val.groupby("srch_id")["price_usd"].transform(
                lambda x: x.rank(pct=True)
            )

        if "position" in val.columns:
            val["log_position"] = np.log(val["position"].clip(lower=1))

        # Hotel-level controls only — search-level features auto-detected
        # and dropped inside the function
        HOTEL_LEVEL_CONTROLS = [f for f in [
            "price_rel",            # within-search price percentile — hotel level
            "is_best_star",         # hotel level
            "is_best_review",       # hotel level
            "is_cheapest_in_srch",  # hotel level
            "prop_brand_bool",      # hotel level
        ] if f in val.columns]

        print("\n" + "="*70)
        print("PROMOTION FLAG BIAS ANALYSIS")
        print("="*70)

        # ------------------------------------------------------------------
        # Check 1 — Conditional logistic regression, no position control
        # ------------------------------------------------------------------
        print("\n--- Check 1 (Data level): Within-search effect, no position control ---")

        results_no_pos, model_raw_no_pos, model_adj_no_pos = promotion_regression_analysis(
            val              = val,
            control_features = HOTEL_LEVEL_CONTROLS,
            target_col       = "promotion_flag",
            outcome_col      = "booking_bool",
        )

        # ------------------------------------------------------------------
        # Check 1b — With log_position control
        # ------------------------------------------------------------------
        if "log_position" in val.columns:
            print("\n--- Check 1b (Data level): Within-search effect, log position control ---")
            print("  log_position used per cascade click model (Craswell et al. 2008)")

            results_with_pos, model_raw_with_pos, model_adj_with_pos = promotion_regression_analysis(
                val              = val,
                control_features = HOTEL_LEVEL_CONTROLS + ["log_position"],
                target_col       = "promotion_flag",
                outcome_col      = "booking_bool",
            )
        else:
            results_with_pos = None
            print("\n  [Skipped Check 1b] position column not found in val.")

        # ------------------------------------------------------------------
        # Check 1c — Robustness: random sort searches only
        # Position is exogenous by design in these searches —
        # closest available estimate to a causal effect of promotion
        # ------------------------------------------------------------------
        if "random_bool" in val.columns:
            val_random = val[val["random_bool"] == 1].copy()
            print(f"\n--- Check 1c (Robustness): Random sort searches only ---")
            print(f"  Position exogenous by design — Expedia randomised display order")
            print(f"  This is the cleanest available estimate of the genuine promotion effect")
            print(f"  N searches (random only): {val_random['srch_id'].nunique():,}")

            controls_random = HOTEL_LEVEL_CONTROLS.copy()
            if "log_position" in val_random.columns:
                controls_random = controls_random + ["log_position"]

            results_random, _, _ = promotion_regression_analysis(
                val              = val_random,
                control_features = controls_random,
                target_col       = "promotion_flag",
                outcome_col      = "booking_bool",
            )
        else:
            results_random = None
            print("\n  [Skipped Check 1c] random_bool not found in val.")

        # ------------------------------------------------------------------
        # Check 2a — Model level: within-search rank gap
        # ------------------------------------------------------------------
        print("\n--- Check 2a (Model level): Within-search rank gap ---")
        print("  Measures model behaviour — independent of regression checks above")

        rank_result = compute_rank_bias(
            val, knn_base, KNN_FEATURES,
            alpha=0.0, target_col="promotion_flag"
        )
        rank_bias_report(
            rank_result,
            stage      = "KNN Baseline",
            target_col = "promotion_flag"
        )

        # ------------------------------------------------------------------
        # Synthesis
        # ------------------------------------------------------------------
        print("\n--- Synthesis ---")

        def get_or(results_df, feature, model_label):
            """Safely extract OR and p-value from results dataframe."""
            row = results_df[
                (results_df["feature"] == feature) &
                (results_df["model"]   == model_label)
            ]
            if row.empty:
                return None, None
            return row.iloc[0]["OR"], row.iloc[0]["p_value"]

        or_no_pos,   p_no_pos   = get_or(results_no_pos,   "promotion_flag", "Adjusted")
        or_raw,      p_raw      = get_or(results_no_pos,   "promotion_flag", "Unadjusted")
        rank_gap = rank_result["rank_gap"].mean() if not rank_result.empty else None

        print(f"\n  {'Metric':<55} {'Value':>12}")
        print(f"  {'-'*70}")
        print(f"  {'Within-search OR — unadjusted':<55} {or_raw:>12.3f}")
        print(f"  {'Within-search OR — adjusted (no position)':<55} {or_no_pos:>12.3f}")

        if results_with_pos is not None:
            or_with_pos, p_with_pos = get_or(results_with_pos, "promotion_flag", "Adjusted")
            attenuation_pos = (1 - or_with_pos / or_no_pos) * 100
            print(f"  {'Within-search OR — adjusted (log position)':<55} "
                f"{or_with_pos:>12.3f}")
            print(f"  {'OR attenuation from log position':<55} "
                f"{attenuation_pos:>11.1f}%")
        else:
            or_with_pos      = None
            attenuation_pos  = None

        if results_random is not None:
            or_random, p_random = get_or(results_random, "promotion_flag", "Adjusted")
            print(f"  {'Within-search OR — random sort only (robustness)':<55} "
                f"{or_random:>12.3f}")
        else:
            or_random = None

        if rank_gap is not None:
            print(f"  {'Model rank gap (KNN)':<55} {rank_gap:>+12.4f}  "
                f"(d=0.86, large)")

        # Verdict
        print(f"\n  Verdict:")

        data_signal  = or_no_pos is not None and or_no_pos > 1 and p_no_pos < 0.05
        model_signal = rank_gap is not None and abs(rank_gap) > 1.0

        if attenuation_pos is not None:
            if attenuation_pos > 30 and model_signal:
                print(f"  Log position explains {attenuation_pos:.1f}% of the within-search "
                    f"promotion effect,")
                print(f"  confirming substantial position bias absorption in the training data.")
                print(f"  The KNN rank gap of {rank_gap:+.4f} substantially exceeds what the")
                print(f"  position-adjusted OR of {or_with_pos:.3f} justifies — over-learning present.")
                print(f"  Mitigation requires explicit position control, addressed in LightGBM.")
            elif attenuation_pos > 10 and model_signal:
                print(f"  Log position explains {attenuation_pos:.1f}% of the within-search "
                    f"promotion effect.")
                print(f"  Partial position bias present. Model rank gap exceeds adjusted "
                    f"data signal.")
            elif data_signal and model_signal:
                print(f"  Promotion effect robust to position control.")
                print(f"  Model rank gap is proportionate to genuine data signal.")
            else:
                print(f"  No meaningful bias detected at either level.")
        else:
            if data_signal and model_signal:
                print(f"  Promotion predicts booking within search and model ranks promoted")
                print(f"  hotels substantially higher. Without position control, cannot")
                print(f"  determine how much reflects genuine signal vs position bias.")
                print(f"  This will be resolved in LightGBM.")
            elif not data_signal and model_signal:
                print(f"  Model over-ranks promoted hotels with no within-search justification.")
                print(f"  Strong evidence of absorbed position bias.")
            else:
                print(f"  No meaningful bias detected at either level.")

        if or_random is not None and or_with_pos is not None:
            random_vs_adjusted = abs(or_random - or_with_pos) / or_with_pos * 100
            print(f"\n  Robustness check:")
            print(f"  OR in random sort searches ({or_random:.3f}) vs position-adjusted "
                f"full sample ({or_with_pos:.3f})")
            print(f"  Difference: {random_vs_adjusted:.1f}%")
            print(f"  {'Results are consistent — position adjustment is adequate' if random_vs_adjusted < 10 else 'Gap remains between random-sort and position-adjusted estimates — residual position confounding likely'}")

    else:
        print("\n  [Skipped] promotion_flag not found in val.")


    for feat in bias_features:
        if feat not in val.columns:
            continue

        print(f"\n{'#'*60}")
        print(f"BIAS EVALUATION FOR FEATURE: {feat}")
        print(f"{'#'*60}")

        # Pre-compute scores once per feature
        tmp = val.copy()
        tmp["score"] = _score_model(knn_base, val, KNN_FEATURES, alpha=0.0, target_col=None)

        # Check A: NDCG@5 grouped by what the user actually booked
        booked_label = (val[val["booking_bool"] == 1][["srch_id", feat]]
                        .drop_duplicates("srch_id")
                        .rename(columns={feat: f"booked_{feat}"}))

        ndcg_rows = []
        for srch_id, grp in tmp.groupby("srch_id"):
            if grp["relevance"].sum() == 0:
                continue
            s = ndcg_score(
                grp["relevance"].values.reshape(1, -1),
                grp["score"].values.reshape(1, -1),
                k=5,
            )
            ndcg_rows.append({"srch_id": srch_id, "ndcg5": s})

        feat_ndcg = pd.DataFrame(ndcg_rows).merge(booked_label, on="srch_id", how="left")

        print(f"\n[Check A] NDCG@5 by booked {feat}")
        try:
            bias_report(feat_ndcg, group_col=f"booked_{feat}",
                        group_labels={1: f"Booked {feat}=1", 0: f"Booked {feat}=0"},
                        stage=f"KNN Baseline — {feat}")
        except Exception as e:
            print(f"  Skipped Check A: {e}")

        # Check B: within-search rank gap
        print(f"\n[Check B] Within-search rank gap for {feat}")
        try:
            rank_result = compute_rank_bias(val, knn_base, KNN_FEATURES, target_col=feat)
            rank_bias_report(rank_result, stage=f"KNN Baseline — {feat}", target_col=feat)
        except Exception as e:
            print(f"  Skipped Check B: {e}")

    '''# =========================================================================
    # POST-PROCESSING MITIGATION — SCORE ADJUSTMENT (PROMOTION FLAG)
    # =========================================================================

    print("\n" + "="*70)
    print("POST-PROCESSING MITIGATION — ALPHA CALIBRATION")
    print("="*70)

    MITIGATION_COL = "promotion_flag"  # column being mitigated
                                        # alpha bonus applied to value=0 (non-promoted)

    # Inspect score distribution to set a sensible alpha grid
    _scores = _score_model(knn_base, val, KNN_FEATURES, alpha=0.0, target_col=None)
    print(f"\n  Model score distribution (val set):")
    print(f"    min:  {_scores.min():.6f}")
    print(f"    p25:  {np.percentile(_scores, 25):.6f}")
    print(f"    mean: {_scores.mean():.6f}")
    print(f"    p75:  {np.percentile(_scores, 75):.6f}")
    print(f"    max:  {_scores.max():.6f}")
    iqr = np.percentile(_scores, 75) - np.percentile(_scores, 25)
    print(f"    IQR:  {iqr:.6f}")

    # Anchor alpha grid to smallest non-zero score
    _nonzero_scores   = _scores[_scores > 0]
    score_min_nonzero = float(_nonzero_scores.min()) if len(_nonzero_scores) > 0 else 1e-6
    print(f"    Non-zero score min: {score_min_nonzero:.8f}")

    alpha_grid = [round(score_min_nonzero * pct, 10)
                for pct in [0.001, 0.005, 0.01, 0.05,
                            0.10,  0.25,  0.50, 0.75, 1.0, 2.0]]

    # ------------------------------------------------------------------
    # Baseline NDCG@5 (alpha=0)
    # ------------------------------------------------------------------
    ndcg_baseline_df = compute_ndcg_by_group(val, knn_base, KNN_FEATURES,
                                            group_col=f"booked_{MITIGATION_COL}",
                                            alpha=0.0)
    ndcg_baseline = ndcg_baseline_df["ndcg5"].mean()

    print(f"\n  {'Alpha':<12} {'Rank gap':>12} {'NDCG@5':>10} {'ΔNDCG':>10}")
    print(f"  {'-'*50}")
    print(f"  {'0.000 (base)':<12} {'—':>12} {ndcg_baseline:>10.4f} {'0.0000':>10}  ← baseline")

    # ------------------------------------------------------------------
    # Grid search over alpha
    # ------------------------------------------------------------------
    alpha_results = []

    for alpha in alpha_grid:
        # Rank gap with adjusted scores
        rank_df = compute_rank_bias(val, knn_base, KNN_FEATURES,
                                    alpha=alpha, target_col=MITIGATION_COL)

        if rank_df.empty or "rank_gap" not in rank_df.columns:
            continue

        rank_gap = rank_df["rank_gap"].mean()

        # NDCG with adjusted scores
        ndcg_df   = compute_ndcg_by_group(val, knn_base, KNN_FEATURES,
                                        group_col=f"booked_{MITIGATION_COL}",
                                        alpha=alpha)
        ndcg_val  = ndcg_df["ndcg5"].mean()
        delta     = ndcg_val - ndcg_baseline

        alpha_results.append({
            "alpha":    alpha,
            "rank_gap": rank_gap,
            "ndcg":     ndcg_val,
            "delta":    delta,
        })

        print(f"  {alpha:<12.6f} {rank_gap:>+12.4f} {ndcg_val:>10.4f} {delta:>+10.4f}")

    # ------------------------------------------------------------------
    # Select best alpha — smallest |rank gap| within NDCG tolerance
    # ------------------------------------------------------------------
    NDCG_TOLERANCE = 0.005  # maximum acceptable NDCG drop

    if alpha_results:
        results_df   = pd.DataFrame(alpha_results)
        within_tol   = results_df[results_df["delta"] >= -NDCG_TOLERANCE]

        if not within_tol.empty:
            best_idx   = within_tol["rank_gap"].abs().idxmin()
            best_alpha = within_tol.loc[best_idx, "alpha"]
            best_gap   = within_tol.loc[best_idx, "rank_gap"]
            best_ndcg  = within_tol.loc[best_idx, "ndcg"]
            print(f"\n  Best alpha within NDCG tolerance (±{NDCG_TOLERANCE}): "
                f"{best_alpha:.6f}")
            print(f"  Rank gap at best alpha: {best_gap:+.4f}")
            print(f"  NDCG@5 at best alpha:   {best_ndcg:.4f}  "
                f"(Δ={best_ndcg - ndcg_baseline:+.4f})")
        else:
            # All alphas exceed tolerance — pick least damaging
            best_idx   = results_df["rank_gap"].abs().idxmin()
            best_alpha = results_df.loc[best_idx, "alpha"]
            best_gap   = results_df.loc[best_idx, "rank_gap"]
            best_ndcg  = results_df.loc[best_idx, "ndcg"]
            print(f"\n  No alpha within NDCG tolerance — selecting least damaging.")
            print(f"  Best alpha: {best_alpha:.6f} | Rank gap: {best_gap:+.4f} | "
                f"NDCG: {best_ndcg:.4f} (Δ={best_ndcg - ndcg_baseline:+.4f})")
            print(f"  Limitation: mitigation overshoots — NDCG cost exceeds tolerance.")
            print(f"  This is a known limitation of post-processing on KNN and will")
            print(f"  be addressed in the LightGBM model.")
    else:
        best_alpha = 0.0
        print("\n  No valid alpha results — defaulting to no mitigation.")

    # ------------------------------------------------------------------
    # Re-run Check B (rank gap) with best alpha
    # ------------------------------------------------------------------
    print(f"\n--- Rank gap after mitigation (alpha={best_alpha:.6f}) ---")
    rank_result_fair = compute_rank_bias(val, knn_base, KNN_FEATURES,
                                        alpha=best_alpha,
                                        target_col=MITIGATION_COL)
    rank_bias_report(rank_result_fair,
                    stage=f"After mitigation (alpha={best_alpha:.6f})",
                    target_col=MITIGATION_COL)

    # ------------------------------------------------------------------
    # Re-run Check A (NDCG) with best alpha
    # ------------------------------------------------------------------
    print(f"\n--- NDCG@5 after mitigation (alpha={best_alpha:.6f}) ---")

    tmp_adj = val.copy()
    tmp_adj["score"] = _score_model(knn_base, val, KNN_FEATURES,
                                    alpha=best_alpha,
                                    target_col=MITIGATION_COL)

    booked_label_adj = (val[val["booking_bool"] == 1][["srch_id", MITIGATION_COL]]
                        .drop_duplicates("srch_id")
                        .rename(columns={MITIGATION_COL: f"booked_{MITIGATION_COL}"}))

    ndcg_rows_adj = []
    for srch_id, grp in tmp_adj.groupby("srch_id"):
        if grp["relevance"].sum() == 0:
            continue
        s = ndcg_score(
            grp["relevance"].values.reshape(1, -1),
            grp["score"].values.reshape(1, -1),
            k=5,
        )
        ndcg_rows_adj.append({"srch_id": srch_id, "ndcg5": s})

    ndcg_fair_df = (pd.DataFrame(ndcg_rows_adj)
                    .merge(booked_label_adj, on="srch_id", how="left"))

    bias_report(ndcg_fair_df,
                group_col=f"booked_{MITIGATION_COL}",
                group_labels={1: "Booked promoted", 0: "Booked non-promoted"},
                stage=f"After mitigation (alpha={best_alpha:.6f})")

    # ------------------------------------------------------------------
    # Mitigation summary table
    # ------------------------------------------------------------------
    rank_gap_before = compute_rank_bias(val, knn_base, KNN_FEATURES,
                                        alpha=0.0,
                                        target_col=MITIGATION_COL)["rank_gap"].mean()
    rank_gap_after  = rank_result_fair["rank_gap"].mean()
    ndcg_after      = ndcg_fair_df["ndcg5"].mean()

    print(f"\n--- Mitigation Summary ---")
    print(f"  {'Metric':<45} {'Before':>10} {'After':>10} {'Δ':>10}")
    print(f"  {'-'*75}")
    print(f"  {'Overall NDCG@5':<45} {ndcg_baseline:>10.4f} {ndcg_after:>10.4f} "
        f"{ndcg_after - ndcg_baseline:>+10.4f}")
    print(f"  {'Mean rank gap (promoted − non-promoted)':<45} "
        f"{rank_gap_before:>+10.4f} {rank_gap_after:>+10.4f} "
        f"{rank_gap_after - rank_gap_before:>+10.4f}")
    print(f"  {'Alpha applied':<45} {'0.000':>10} {best_alpha:>10.6f}")
    print(f"\n  Mitigation effective if:")
    print(f"    - rank gap moves toward 0  (positional bias reduced)")
    print(f"    - NDCG@5 drop < {NDCG_TOLERANCE}         (ranking quality preserved)")

    within_tolerance = (ndcg_after - ndcg_baseline) >= -NDCG_TOLERANCE
    gap_reduced      = abs(rank_gap_after) < abs(rank_gap_before)

    if gap_reduced and within_tolerance:
        print(f"\n  Verdict: mitigation successful — bias reduced within NDCG tolerance.")
    elif gap_reduced and not within_tolerance:
        print(f"\n  Verdict: bias reduced but NDCG cost exceeds tolerance.")
        print(f"  Limitation: post-processing alpha is a blunt instrument on KNN.")
        print(f"  A more targeted mitigation will be applied in LightGBM.")
    else:
        print(f"\n  Verdict: mitigation ineffective — alpha does not reduce rank gap.")
        print(f"  This confirms post-processing is insufficient for this bias type.")
        print(f"  Root cause mitigation requires LightGBM with position control.")'''

    # -------------------------------------------------------------------------
    # Save model artefact
    # -------------------------------------------------------------------------
    date_str = datetime.now().strftime("%Y%m%d")
    try:
        joblib.dump(knn_base,
                    f"{MODELS_DIR}/knn_base_{date_str}.pkl")
        joblib.dump(KNN_FEATURES,
                    f"{MODELS_DIR}/knn_features_{date_str}.pkl")
        print(f"  Artefacts saved to {MODELS_DIR}/")
    except OSError as e:
        print(f"  Could not save artefacts (disk quota): {e}")
        print("  Submission CSV is already saved — that is all you need.")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    print("\nGenerating submission...")
    make_submission(test, knn_base, model_name="knn-base",
                    features=KNN_FEATURES)


if __name__ == "__main__":
    main()
