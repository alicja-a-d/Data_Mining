# data_mining

Overleaf link: https://www.overleaf.com/4662494197drnchxjqfhfn#9b3bdd



ML_WORKFLOW/
├── 1. DATA PARTITIONING
│   ├── Train Set (80%) ───────────────┐
│   └── Test Set (20%; locked away)─┐  │
│                                   │  │
├── 2. BASELINE EVALUATION             │
│   └── Model: XGBoost (Default) <─────┘
│       └── Goal: to establish baseline performance
│
├── 3. HYPERPARAMETER OPTIMIZATION
│   └── RandomizedSearchCV (on Train Set)
│       ├── Cross-Validation Strategy: TimeSeriesSplit (k=5)
│       ├── Metric: Mean F1-Weighted
│       └── Output: Best_Params_
│
├── 4. STABILITY VALIDATION
│   └── 5-Fold TimeSeriesSplit (using best parameters )
│       └── Goal: check variance across temporal windows
│
├── 5. MODEL FINALIZATION
│   └── Retrain the model (full train set on best parameters)
│
└── 6. FINAL BENCHMARK
    └── Evaluation on test Set <───┘
        └── Goal: analyse final performance