# Clinical Trial Retention Intelligence

### An Explainable Machine Learning Framework for Clinical Trial Dropout Prediction and Uncertainty-Aware Analysis

## Project Overview

Clinical trial participant dropout is an important challenge that can affect trial completion, research costs, and the reliability of study outcomes.

**Clinical Trial Retention Intelligence** is a machine learning-based research application designed to estimate clinical trial dropout rates using trial characteristics, participant eligibility information, conditions, interventions, trial summaries, and sponsor information.

The framework combines predictive modelling with uncertainty estimation, selective abstention, explainable AI, and evidence-group analysis to provide a more transparent approach to clinical trial dropout prediction.

## Key Features

### 1. Machine Learning-Based Dropout Prediction
- Predicts estimated clinical trial dropout rates.
- Uses structured trial characteristics and text-based information.
- Uses Ridge Regression as the selected prediction model.
- Provides interactive trial information inputs.

### 2. Uncertainty Estimation
- Implements Conformalized Quantile Regression (CQR).
- Generates lower and upper prediction bounds.
- Calculates prediction interval width.
- Evaluates empirical prediction interval coverage.

### 3. Selective Prediction and Abstention
- Uses prediction interval width to assess prediction uncertainty.
- Applies a calibration-based threshold.
- Identifies predictions that may require additional evidence or expert review.

### 4. Explainable AI Using SHAP
- Provides global feature importance.
- Explains individual model predictions.
- Identifies features contributing positively or negatively to the model output.

### 5. Evidence-Guided Analysis
- Examines the effect of hiding different groups of trial information.
- Compares changes in prediction interval width.
- Displays previously validated evidence-group results.

## Input Features

The application accepts 14 input features:

| Category | Features |
|---|---|
| Trial characteristics | Duration, number of sites, enrollment, start year |
| Trial design | Phase, allocation, masking |
| Participant information | Minimum age, gender, eligibility criteria |
| Medical information | Conditions, drugs/interventions |
| Trial description | Brief summary |
| Sponsorship | Sponsor |

**Target variable:** Clinical trial dropout rate (`dropout_rate`).

## Dashboard Sections

The Streamlit application contains three main sections:

1. **Overview** – Model performance, dataset summary, selective prediction results, conformal prediction validation, and evidence validation.
2. **Predict** – Interactive dropout prediction, uncertainty estimation, abstention assessment, individual SHAP explanation, and downloadable prediction report.
3. **Explainability & Evidence** – Global feature importance, individual evidence-group masking, and validation results.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Ridge Regression
- Conformalized Quantile Regression
- SHAP
- Matplotlib
- Joblib
- Streamlit

## Model Evaluation

The framework evaluates model performance using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Prediction interval coverage
- Prediction interval width
- Selective prediction performance

## Research Considerations and Limitations

- The application is an academic research prototype, not a clinically validated decision-making system.
- Predictions represent model estimates and should not be interpreted as guaranteed dropout rates.
- Conformal prediction provides marginal coverage guarantees under its assumptions; subgroup coverage may differ.
- SHAP explanations describe model behaviour and do not establish causal relationships.
- Evidence-group masking is a model-based experiment and does not establish that collecting particular information will causally reduce uncertainty.
- Additional external validation is necessary before real-world clinical or research use.

## Future Scope

- External validation using independent clinical trial datasets.
- Improved subgroup uncertainty calibration.
- Advanced text representation techniques.
- More robust evidence recommendation strategies.
- Integration with clinical research decision-support workflows.

## Author

Developed as an academic research project in **Data Science and Bio-AI**.

---

**Clinical Trial Retention Intelligence**

*Explainable Machine Learning | Uncertainty Estimation | Selective Prediction | Evidence-Guided Analysis*
