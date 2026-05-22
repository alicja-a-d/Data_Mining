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
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

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


#we keep only features that are related continuous
KNN_FEATURES = [
    # hotel quality
    "prop_starrating", "prop_review_score", "prop_location_score1",
    "prop_location_score2", "prop_log_historical_price", "prop_brand_bool",
    "promotion_flag",
    # price signals
    "price_usd", "price_ratio", "price_vs_prop_mean", "price_vs_prop_median",
    "price_vs_min", "price_vs_hist", "price_pct_rank",
    # search context
    "srch_length_of_stay", "srch_booking_window", "srch_adults_count",
    "srch_children_count", "srch_room_count", "srch_saturday_night_bool",
    # within-search comparisons (these are the most useful for KNN)
    "review_vs_max", "star_vs_max", "loc1_vs_max",
    "is_cheapest_in_srch", "is_best_review", "is_best_star",
    "starrating_pct_rank", "review_pct_rank", "loc1_rank",
    # user match
    "star_vs_hist", "quality_score", "bool_same_country",
    # context
    "is_last_minute", "srch_n_hotels",
    "has_review_score", "has_loc_score2",
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
    Imputation strategy:
    - Add binary flags for every column with informative missingness.
    - visitor_hist_*: leave NaN (null = new customer).
    - srch_query_affinity_score: fill 0 (null = not indexed online).
    - prop_review_score: leave NaN — LightGBM routes NaN rows to the optimal
      branch; flag has_review_score captures the pattern explicitly.
    - orig_destination_distance: leave NaN for same reason.
    - prop_location_score2: fill via destination-grouped median (needed for
      within-search rank features computed later).
    - All medians computed from train_df only to avoid leakage.
    """
    if train_df is None:
        train_df = data

    # Visitor history: null = new customer; leave NaN for LightGBM
    data["new_customer"]            = data["visitor_hist_starrating"].isnull().astype("int8")

    # Review score: null != 0 per dataset spec; leave NaN for LightGBM
    data["has_review_score"] = data["prop_review_score"].notnull().astype("int8")

    # Location score 2: destination-grouped median, needed for rank features
    data["has_loc_score2"] = data["prop_location_score2"].notnull().astype("int8")
    dest_loc2 = train_df.groupby("srch_destination_id")["prop_location_score2"].median()
    data["prop_location_score2"] = (
        data["prop_location_score2"]
        .fillna(data["srch_destination_id"].map(dest_loc2))
        .fillna(train_df["prop_location_score2"].median())
    )

    # Query affinity: null = not in internet searches; leave NaN for LightGBM
    data["has_query_affinity"]        = data["srch_query_affinity_score"].notnull().astype("int8")

    # Distance: leave NaN for LightGBM
    data["has_distance"] = data["orig_destination_distance"].notnull().astype("int8")

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
 
    Computes:
    - Property price stats from all data (no labels — no leakage risk).
    - Estimated position from non-random training displays (train_df only).
    - Within-search relative features: ranks, percentile ranks,
      best-in-search comparisons.
    - User-hotel match features (masked to 0 for new customers).
    - Quality interaction features and context flags.
    """
    if train_df is None:
        train_df = data
 
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
 
    # --- Property-level price stats - CHANGE - ONLY using (no labels — safe to use all data) -------
    # Using all_data so test hotels not in train still get a price history
    all_data    = pd.concat([train_df, data], ignore_index=True)
    price_stats = (all_data.groupby("prop_id")
                   .agg(prop_mean_price   = ("price_usd", "mean"),
                        prop_std_price    = ("price_usd", "std"),
                        prop_median_price = ("price_usd", "median"))
                   .reset_index())
 
    # --- Estimated position from non-random training displays (train only) ---
    # Inverted so higher = historically shown higher = stronger prior signal
    estimated_pos = (train_df[train_df["random_bool"] == 0]
                     .groupby(["srch_destination_id", "prop_id"])["position"]
                     .mean()
                     .reset_index()
                     .rename(columns={"position": "estimated_position"}))
    estimated_pos["estimated_position"] = 1.0 / estimated_pos["estimated_position"]
 
    # Merge
    data = (data
            .merge(price_stats,   on="prop_id",                          how="left")
            .merge(estimated_pos, on=["srch_destination_id", "prop_id"], how="left"))
 
    # Fallback fills
    data["prop_std_price"]     = data["prop_std_price"].fillna(0)
    data["estimated_position"] = data["estimated_position"].fillna(0)
 
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
    data["loc1_rank"]       = g["prop_location_score1"].transform(
                                  "rank", ascending=False, method="dense")
    data["loc2_rank"]       = g["prop_location_score2"].transform(
                                  "rank", ascending=False, method="dense")
 
    # Percentile ranks (0-1): comparable across searches of different sizes
    data["price_pct_rank"]      = g["price_usd"].transform("rank", pct=True)
    data["starrating_pct_rank"] = g["prop_starrating"].transform(
                                      "rank", ascending=False, pct=True)
    data["review_pct_rank"]     = g["prop_review_score"].transform(
                                      "rank", ascending=False, pct=True)
 
    # --- User-hotel match (masked to 0 for new customers) --------------------
    data["star_vs_hist"]  = np.where(
        data["new_customer"] == 1, 0,
        data["prop_starrating"] - data["visitor_hist_starrating"]
    )
    data["price_vs_hist"] = np.where(
        data["new_customer"] == 1, 0,
        data["price_usd"] - data["visitor_hist_adr_usd"]
    )
 
    # --- Quality interactions ------------------------------------------------
    data["ad_vs_real"]              = (data["prop_starrating"]
                                       - data["prop_review_score"].fillna(0))
    data["quality_score"]           = (data["prop_starrating"]
                                       * data["prop_review_score"].fillna(0))
    data["location_score_combined"] = (data["prop_location_score1"] * 0.7 +
                                       data["prop_location_score2"] * 0.3)
 
    # --- Context flags -------------------------------------------------------
    data["bool_same_country"] = (
        data["visitor_location_country_id"] == data["prop_country_id"]
    ).astype("int8")
    data["is_last_minute"] = (data["srch_booking_window"] <= 3).astype("int8")
    data["is_long_stay"]   = (data["srch_length_of_stay"] >= 7).astype("int8")
 
    # --- Within-search best comparisons --------------------------------------
    # How far is each hotel from the best option in the same search?
    data["price_vs_min"]  = data["price_usd"] - g["price_usd"].transform("min")
    data["review_vs_max"] = (g["prop_review_score"].transform("max")
                             - data["prop_review_score"].fillna(0))
    data["star_vs_max"]   = (g["prop_starrating"].transform("max")
                             - data["prop_starrating"])
    data["loc1_vs_max"]   = (g["prop_location_score1"].transform("max")
                             - data["prop_location_score1"])
 
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
# 10. MODEL: LIGHTGBM LAMBDAMART
# =============================================================================


def train_knn(train: pd.DataFrame,
              val:   pd.DataFrame,
              features: list,
              sample_weight: np.ndarray = None):
    """
    Train a KNN ranking model with optional sample re-weighting for bias mitigation.

    Parameters
    ----------
    train         : training DataFrame (already split, features engineered)
    val           : validation DataFrame
    features      : list of feature column names (no srch_id)
    sample_weight : per-row weights aligned with train index (from compute_sample_weights)
                    If None, all rows are weighted equally.

    Returns
    -------
    best_model    : fitted Pipeline (scaler + knn) with best hyperparameters
    ndcg_val      : NDCG@5 on the validation set
    """

    # ------------------------------------------------------------------
    # Sample to make CV feasible (sample by srch_id to keep searches intact)
    # ------------------------------------------------------------------
    unique_ids  = train["srch_id"].unique()
    sample_ids  = pd.Series(unique_ids).sample(
        n=min(5000, len(unique_ids)), random_state=SEED
    )
    df_sample   = train[train["srch_id"].isin(sample_ids)].copy()

    # Align sample weights to the sampled rows
    if sample_weight is not None:
        sw_series  = pd.Series(sample_weight, index=train.index)
        sw_sample  = sw_series[df_sample.index].values
    else:
        sw_sample  = None

    X_train     = df_sample[features]
    y_train     = df_sample["relevance"]
    groups      = df_sample["srch_id"]

    X_val       = val[features]
    y_val       = val["relevance"]

    # ------------------------------------------------------------------
    # NDCG@5 scorer — operates at the search level, not row level
    # Uses predicted probabilities: higher prob of class 2 (booked) → higher rank
    # ------------------------------------------------------------------
    def ndcg_scorer_fn(estimator, X, y):
        """
        Custom scorer that computes mean NDCG@5 across searches.
        Uses the probability of the highest relevance class as the ranking score.
        """
        # Reconstruct srch_id alignment via positional index in df_sample
        # (GroupKFold preserves group structure so we can use groups directly)
        proba       = estimator.predict_proba(X)          # shape (n, n_classes)
        classes     = estimator.named_steps["knn"].classes_
        # Score = probability of being booked (class 2) + 0.1 * prob of clicked (class 1)
        # This gives a richer ranking signal than just argmax
        score_vec   = np.zeros(len(X))
        if 2 in classes:
            score_vec += proba[:, list(classes).index(2)] * 1.0
        if 1 in classes:
            score_vec += proba[:, list(classes).index(1)] * 0.1

        # Compute per-search NDCG@5
        # We need srch_id — attach it back via the index
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
                    k=5
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
        "knn__n_neighbors": [10, 20, 50],
        "knn__weights":     ["distance"],
        "knn__metric":      ["manhattan"],
    }

    # GroupKFold: keep all rows of a search in the same fold
    cv_strategy = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(
        pipeline,
        param_grid  = param_grid,
        cv          = cv_strategy,
        scoring     = ndcg_scorer_fn,
        n_jobs      = -1,
        verbose     = 2,
        refit       = True,
    )

    print("\nStarting KNN Cross-Validation Grid Search...")
    # Pass sample_weight through pipeline to knn step
    fit_params = {}
    if sw_sample is not None:
        fit_params["knn__sample_weight"] = sw_sample

    grid_search.fit(X_train, y_train, groups=groups, **fit_params)

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


# =============================================================================
# 11 FEATURE SELECTION BY IMPORTANCE
# =============================================================================

def select_features(model, features: list,
                    threshold_pct: float = 0.5) -> list:
    """
    Drop features that contribute less than threshold_pct% of total
    importance gain. Prints a full importance table and saves a plot.

    threshold_pct=0.5 means: drop any feature below 0.5% of total gain.
    Typical outcome: keeps ~30-45 features from a 60-feature candidate set.
    """



    model_feature_names = model.booster_.feature_name()
    importance = pd.DataFrame({
        "feature":    model_feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


   
    total = importance["importance"].sum()
    importance["importance_pct"] = (importance["importance"] / total * 100).round(2)
    importance["cumulative_pct"] = importance["importance_pct"].cumsum().round(2)

    print("\n--- Feature Importance (all features) ---")
    print(importance.to_string(index=False))

    # Plot full importance
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.25)))
    colors = ["#378ADD" if p >= threshold_pct else "#D85A30"
              for p in importance["importance_pct"][::-1]]
    ax.barh(importance["feature"][::-1],
            importance["importance"][::-1],
            color=colors, edgecolor="white")
    ax.axvline(total * threshold_pct / 100, color="#D85A30",
               linestyle="--", linewidth=1,
               label=f"Threshold ({threshold_pct}% of total)")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance — blue=kept, red=dropped")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/lgb_feature_importance_full.png", dpi=150)
    plt.show()

    # Select features above threshold
    kept    = importance[importance["importance_pct"] >= threshold_pct]["feature"].tolist()
    dropped = importance[importance["importance_pct"] <  threshold_pct]["feature"].tolist()

    # Added:
    if len(kept) == 0:
        print(f"  WARNING: threshold_pct={threshold_pct} dropped all features. "
              f"Keeping all {len(model_feature_names)} features.")
        kept = model_feature_names
        dropped = []

    print(f"\n  Kept   {len(kept)} features  (>= {threshold_pct}% importance)")
    print(f"  Dropped {len(dropped)} features:")
    for f in dropped:
        row = importance[importance["feature"] == f].iloc[0]
        print(f"    {f:<40} {row['importance_pct']:.2f}%")

    return kept


# =============================================================================
# 11b. RETRAIN ON FULL DATA
# =============================================================================

def retrain_on_full_data(df_full: pd.DataFrame, best_iteration: int,
                         features: list = None):
    """
    Retrain on 100% of labelled data at the iteration count found by early
    stopping. Uses the trimmed feature list if provided.
    """
    if features is None:
        features = FEATURES

    # Scale up iteration count slightly: with 100% of data the model has
    # more signal per tree and can usefully run a bit longer than on 80%
    scaled_iteration = int(best_iteration * 1.15)
    print(f"\n--- Retraining on full data ({len(features)} features, "
          f"{scaled_iteration} iterations [={best_iteration}*1.15]) ---")

    df_s = df_full.sort_values("srch_id").reset_index(drop=True)

    missing = [f for f in features if f not in df_s.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    active_cats = [c for c in CAT_FEATURES if c in features]

    X      = df_s[features]
    y      = df_s["relevance"]
    groups = df_s.groupby("srch_id").size().values

    model = lgb.LGBMRanker(
        objective          = "lambdarank",
        metric             = "ndcg",
        n_estimators       = scaled_iteration,
        learning_rate      = 0.01,
        num_leaves         = 255,
        min_child_samples  = 100,
        min_data_per_group = 50,
        subsample          = 0.8,
        subsample_freq     = 1,
        colsample_bytree   = 0.7,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        label_gain         = [0, 1, 5],
        random_state       = SEED,
        n_jobs             = -1,
        verbose            = -1,
    )

    model.fit(
        X, y,
        group               = groups,
        categorical_feature = active_cats,
        callbacks           = [lgb.log_evaluation(period=100)],
    )

    print("  Done.")
    return model


# =============================================================================
# 12. SUBMISSION GENERATION
# =============================================================================

def make_submission(test: pd.DataFrame, model,
                    model_name: str = "lgb",
                    features: list = None) -> pd.DataFrame:
    if features is None:
        features = FEATURES

    test_s = test.sort_values("srch_id").reset_index(drop=True)

    if isinstance(model, lgb.LGBMRanker):
        test_s["score"] = model.predict(test_s[features])
    else:
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
# 13. MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_path = os.path.join(DATA_DIR, "training_set_VU_DM.csv")
    test_path  = os.path.join(DATA_DIR, "test_set_VU_DM.csv")
    #train_path = os.path.join(DATA_DIR, "sampled_training_set.csv")
    #test_path  = os.path.join(DATA_DIR, "sampled_test_set.csv")
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

    # -------------------------------------------------------------------------
    # Relevance labels
    # -------------------------------------------------------------------------
    train = add_relevance(train)
    val   = add_relevance(val)

    # -------------------------------------------------------------------------
    # Stage 1: LightGBM with ALL features
    # Used to measure importance 
    # -------------------------------------------------------------------------
    print("\n=== STAGE 1: Train with all features to measure importance ===")
    lgb_stage1, _, ndcg_stage1 = train_lightgbm(train, val, features=FEATURES)
    print(f"  Stage 1 NDCG@5: {ndcg_stage1:.4f}")

    # -------------------------------------------------------------------------
    # Feature selection
    # Drop features below 0.5% of total importance gain.
    # Adjust threshold_pct up (e.g. 1.0) to be more aggressive,
    # or down (e.g. 0.2) to keep more features.
    # -------------------------------------------------------------------------
    # Only drop features with truly zero importance (confirmed noise)
    selected_features = select_features(lgb_stage1, FEATURES, threshold_pct=0.05)

    # -------------------------------------------------------------------------
    # Stage 2: Retrain on 100% of data with cleaned feature set
    # -------------------------------------------------------------------------
    print("\nPreparing full dataset for final retrain...")
    df_full = add_features(df, train_df=df)
    df_full = add_relevance(df_full)
    print("\n=== STAGE 2: Retrain on 100% with most relevant features ===")
    final_model = retrain_on_full_data(
        df_full,
        best_iteration = lgb_stage1.best_iteration_,
        features       = selected_features,
    )
     # Save final model
    date_str = datetime.now().strftime("%Y%m%d")
    try:
        # Models
        lgb_stage1.booster_.save_model(f"{MODELS_DIR}/lgb_stage1.txt")
        final_model.booster_.save_model(f"{MODELS_DIR}/lgb_final.txt")
    
        # Feature list
        joblib.dump(selected_features,
                    f"{MODELS_DIR}/selected_features_{date_str}.pkl")
    
        # Stats needed to reproduce feature engineering on new data
        joblib.dump({
            "prop_price_stats":    price_stats,       # prop_mean/std/median_price
            "estimated_positions": estimated_pos,     # estimated_position lookup
            "dest_loc2_medians":   dest_loc2,         # prop_location_score2 fill
            "global_loc2_median":  train_df["prop_location_score2"].median(),
        }, f"{MODELS_DIR}/feature_engineering_stats_{date_str}.pkl")
    
        print(f"All artefacts saved to {MODELS_DIR}/")
    except OSError as e:
        print(f"Could not save: {e}")

    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n--- Final Results ---")
    print(f"  LightGBM stage 1 ({len(FEATURES):2d} features):    {ndcg_stage1:.4f}")
    print(f"  Submitting:       ({len(selected_features):2d} features)")
    
    # -------------------------------------------------------------------------
    # Submissions
    # -------------------------------------------------------------------------
    print("\nGenerating submissions...")
    make_submission(test, knn_model,      model_name="knn-fair", features=KNN_FEATURES)
    make_submission(test, final_model, model_name="lgb-full-best",
                    features=selected_features)   # submit this one

    # -------------------------------------------------------------------------
    # Save selected feature list only (tiny file, models are too large)
    # -------------------------------------------------------------------------
    date_str = datetime.now().strftime("%Y%m%d")
    try:
        joblib.dump(selected_features,
                    f"{MODELS_DIR}/selected_features_{date_str}.pkl")
        print(f"\nFeature list saved to {MODELS_DIR}/")
    except OSError as e:
        print(f"\nCould not save models (disk quota): {e}")
        print("Submission CSVs are already saved — that is all you need.")


if __name__ == "__main__":
    main()
