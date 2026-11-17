TARGET = "dropout_rate"

NUMERIC_COLS = [
    "duration_months", "n_sites", "enrollment", "age_min", "start_year"
]
CATEGORICAL_COLS = [
    "phase", "allocation", "masking", "gender",
    "conditions", "sponsor", "drugs"
]
TEXT_COLS = ["criteria", "brief_summary"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + TEXT_COLS
