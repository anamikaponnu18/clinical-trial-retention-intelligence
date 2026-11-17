# Clinical Trial Dropout Project – Separate Code Files

## Dataset
Place `synthetic_dropout_clean.csv` in this folder. It has 20,000 records and 17 columns.
Target: `dropout_rate`.

## Important
The code deliberately excludes `trial_id` and `is_synthetic` from predictors.
All preprocessing is fitted on training data only.
This is a synthetic dataset; results are not clinical validation.

## Run order

First install:
```bash
pip install -r requirements.txt
```

Run these scripts in order:
1. `01_load_and_inspect.py`
2. `preprocessing.py` (module used by other scripts; optional standalone check)
3. `split_data.py` (optional standalone split inspection)
4. `05_train_models.py`
5. `06_evaluation.py`
6. `07_text_ablation.py`
7. `08_uncertainty_cqr.py`
8. `09_selective_abstention.py`
9. `10_shap_explainability.py`
10. `11_evidence_recommendation.py`
11. `app.py` using Streamlit:
```bash
streamlit run app.py
```

## Notes
- The model training script saves `model_artifacts.joblib`.
- CQR saves `model_artifacts_with_uncertainty.joblib`.
- Selective evaluation saves `final_model_artifacts.joblib`.
- SHAP creates `shap_feature_importance.png`.
- The dashboard expects the generated artifacts and evaluation CSV files.

## Research interpretation
CQR intervals target 90% marginal coverage under exchangeability assumptions.
Selective abstention is based on interval width. The retention table reports actual retained fraction and MAE.
The feature-block recommendation is a simulated masking experiment. It is not causal evidence.
Text SHAP values refer to SVD text components, not individual terms.
