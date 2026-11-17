
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Clinical Trial Retention Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"

RAW_COLUMNS = [
    "duration_months",
    "n_sites",
    "enrollment",
    "start_year",
    "phase",
    "allocation",
    "masking",
    "age_min",
    "gender",
    "criteria",
    "conditions",
    "drugs",
    "brief_summary",
    "sponsor"
]

NUMERIC_COLUMNS = [
    "duration_months",
    "n_sites",
    "enrollment",
    "start_year",
    "age_min"
]

EVIDENCE_GROUPS = {
    "Trial Design": [
        "duration_months", "n_sites", "enrollment",
        "start_year", "phase", "allocation", "masking"
    ],
    "Participant Eligibility": [
        "age_min", "gender", "criteria"
    ],
    "Condition / Intervention": [
        "conditions", "drugs"
    ],
    "Trial Summary": [
        "brief_summary"
    ],
    "Sponsor": [
        "sponsor"
    ]
}


# ==========================================================
# LOAD MODEL ARTIFACTS
# ==========================================================

@st.cache_resource
def load_artifacts():

    model_art = joblib.load(
        OUT / "model_artifacts.joblib"
    )

    cqr_art = joblib.load(
        OUT / "cqr_artifacts.joblib"
    )

    return model_art, cqr_art


try:
    art, cqr = load_artifacts()

    model = art["best_model"]
    builder = art["builder"]

    lower_model = cqr["lower_model"]
    upper_model = cqr["upper_model"]

    q_hat = float(cqr["q_hat"])
    alpha = float(cqr["alpha"])

except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def read_csv(filename):

    path = OUT / filename

    if path.exists():
        return pd.read_csv(path)

    return None


def make_interval(A):

    lower_prediction = lower_model.predict(A)
    upper_prediction = upper_model.predict(A)

    lower = (
        np.minimum(lower_prediction, upper_prediction)
        - q_hat
    )

    upper = (
        np.maximum(lower_prediction, upper_prediction)
        + q_hat
    )

    return (
        np.clip(lower, 0, 1),
        np.clip(upper, 0, 1)
    )


def get_width(A):

    lower, upper = make_interval(A)
    return upper - lower


def percentage(value):

    return f"{value * 100:.2f}%"


def get_options(column, fallback):

    X = art["X_train"]

    if isinstance(X, pd.DataFrame) and column in X.columns:

        values = (
            X[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if values:
            return values

    return fallback


def get_default(column, fallback):

    X = art["X_train"]

    if isinstance(X, pd.DataFrame) and column in X.columns:

        values = pd.to_numeric(
            X[column],
            errors="coerce"
        ).dropna()

        if len(values):
            return float(values.median())

    return fallback


# ==========================================================
# CALIBRATION-BASED ABSTENTION THRESHOLD
# ==========================================================

calibration_widths = get_width(art["A_cal"])

width_threshold = float(
    np.quantile(calibration_widths, 0.90)
)


# ==========================================================
# SHAP EXPLANATION
# ==========================================================

@st.cache_resource
def load_shap_explainer():

    A_train = art["A_train"]

    rng = np.random.default_rng(42)

    n = min(100, A_train.shape[0])

    indices = rng.choice(
        A_train.shape[0],
        size=n,
        replace=False
    )

    background = A_train[indices]

    return shap.LinearExplainer(
        model,
        background
    )


def get_shap_explanation(A):

    feature_names = list(art["feature_names"])

    try:

        explainer = load_shap_explainer()

        values = np.asarray(
            explainer.shap_values(A)
        )

        if values.ndim == 2:
            values = values[0]

        values = values.reshape(-1)

        base_value = float(
            np.asarray(
                explainer.expected_value
            ).reshape(-1)[0]
        )

        return values, base_value, feature_names

    except Exception:

        A_train = art["A_train"]

        mean_values = np.asarray(
            A_train.mean(axis=0)
        ).reshape(-1)

        if hasattr(A, "toarray"):
            current_values = A.toarray().reshape(-1)
        else:
            current_values = np.asarray(A).reshape(-1)

        coefficients = np.asarray(
            model.coef_
        ).reshape(-1)

        contributions = (
            current_values - mean_values
        ) * coefficients

        base_value = float(
            np.asarray(model.intercept_).reshape(-1)[0]
            + np.dot(mean_values, coefficients)
        )

        return contributions, base_value, feature_names


# ==========================================================
# EVIDENCE GROUP ANALYSIS
# ==========================================================

def evidence_analysis(trial_df):

    original_A = builder.transform(trial_df)

    original_width = float(
        get_width(original_A)[0]
    )

    results = []

    for group, columns in EVIDENCE_GROUPS.items():

        masked = trial_df.copy()

        for column in columns:

            if column in NUMERIC_COLUMNS:
                masked[column] = np.nan
            else:
                masked[column] = ""

        masked_A = builder.transform(masked)

        masked_width = float(
            get_width(masked_A)[0]
        )

        results.append({
            "Evidence Group": group,
            "Original Width": original_width,
            "Width After Hiding": masked_width,
            "Increase After Hiding": (
                masked_width - original_width
            )
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            "Increase After Hiding",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.title("🧬 Trial Intelligence")

    st.caption("Clinical AI Research Dashboard")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Predict",
            "Explainability & Evidence"
        ],
        key="main_navigation"
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Selected model**")
    st.write(art["best_name"])

    st.write("**Processed features**")
    st.write(len(art["feature_names"]))

    st.write("**Conformal confidence level**")
    st.write(f"{(1 - alpha) * 100:.0f}%")

    st.divider()

    st.caption(
        "Explainable Machine Learning | "
        "Uncertainty Estimation | Evidence Analysis"
    )


# ==========================================================
# MAIN DASHBOARD HEADER
# ==========================================================

st.title("Clinical Trial Retention Intelligence")

st.caption(
    "Explainable machine learning for clinical trial "
    "dropout prediction and uncertainty-aware analysis."
)

st.divider()


# ==========================================================
# SECTION 1: OVERVIEW
# ==========================================================

if page == "Overview":

    st.header("Project Overview")

    st.write(
        """
        This research framework estimates clinical trial dropout
        rates using machine learning. It also investigates
        prediction uncertainty, selective abstention and the
        influence of available trial information.
        """
    )

    # MODEL PERFORMANCE

    A_test = art["A_test"]
    y_test = art["y_test"]

    predictions = model.predict(A_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Selected Model",
        art["best_name"]
    )

    col2.metric(
        "Test MAE",
        f"{mae:.4f}"
    )

    col3.metric(
        "Test RMSE",
        f"{rmse:.4f}"
    )

    col4.metric(
        "Test R²",
        f"{r2:.4f}"
    )

    st.divider()

    # DATASET SUMMARY

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Samples",
        f"{len(art['y_train']):,}"
    )

    col2.metric(
        "Testing Samples",
        f"{len(art['y_test']):,}"
    )

    col3.metric(
        "Processed Features",
        len(art["feature_names"])
    )

    st.divider()

    # SELECTIVE PREDICTION

    st.subheader("Selective Prediction Performance")

    selective = read_csv(
        "selective_prediction_results.csv"
    )

    if selective is not None:
        st.dataframe(
            selective,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Selective prediction results not found.")

    st.divider()

    # CONFORMAL PREDICTION

    st.subheader("Conformal Prediction Validation")

    coverage = read_csv(
        "cqr_overall_coverage.csv"
    )

    coverage_groups = read_csv(
        "cqr_coverage_by_dropout_group.csv"
    )

    if coverage is not None:
        st.write("Overall coverage")
        st.dataframe(
            coverage,
            use_container_width=True,
            hide_index=True
        )

    if coverage_groups is not None:
        st.write("Coverage by dropout group")
        st.dataframe(
            coverage_groups,
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Overall empirical coverage was approximately 80.7% "
        "against the nominal 80% target. However, the low-dropout "
        "subgroup showed substantial undercoverage. "
        "Marginal coverage does not guarantee subgroup coverage."
    )

    st.divider()

    # EVIDENCE VALIDATION

    st.subheader("Evidence Validation Results")

    validation = read_csv(
        "evidence_validation_comparison.csv"
    )

    if validation is not None:
        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Evidence validation results not found.")


# ==========================================================
# SECTION 2: PREDICT
# ==========================================================

elif page == "Predict":

    st.header("Clinical Trial Dropout Prediction")

    st.write(
        "Enter trial characteristics to estimate dropout "
        "and assess the reliability of the prediction."
    )

    presets = {

        "Custom Trial": {},

        "Small Phase 1 Trial": {
            "duration_months": 12,
            "n_sites": 2,
            "enrollment": 40,
            "start_year": 2024,
            "phase": "Phase 1",
            "allocation": "Randomized",
            "masking": "Double",
            "age_min": 18,
            "gender": "All",
            "criteria": "Healthy adults aged 18 to 45",
            "conditions": "Healthy volunteers",
            "drugs": "Investigational drug",
            "brief_summary": "Phase 1 safety and tolerability study",
            "sponsor": "University Research Centre"
        },

        "Large Phase 3 Trial": {
            "duration_months": 36,
            "n_sites": 80,
            "enrollment": 2500,
            "start_year": 2023,
            "phase": "Phase 3",
            "allocation": "Randomized",
            "masking": "Double",
            "age_min": 18,
            "gender": "All",
            "criteria": "Adults with confirmed type 2 diabetes",
            "conditions": "Type 2 Diabetes Mellitus",
            "drugs": "Investigational antidiabetic medication",
            "brief_summary": "Multicentre randomized diabetes treatment study",
            "sponsor": "Global Pharmaceutical Company"
        },

        "Sparse Information Trial": {
            "duration_months": 18,
            "n_sites": 10,
            "enrollment": 150,
            "start_year": 2025,
            "phase": "Phase 2",
            "allocation": "Randomized",
            "masking": "None",
            "age_min": 18,
            "gender": "All",
            "criteria": "",
            "conditions": "Asthma",
            "drugs": "",
            "brief_summary": "",
            "sponsor": ""
        }
    }

    preset_name = st.selectbox(
        "Choose a sample trial",
        list(presets.keys())
    )

    preset = presets[preset_name]

    st.subheader("Trial Characteristics")

    with st.form("trial_form"):

        col1, col2 = st.columns(2)

        with col1:

            duration = st.number_input(
                "Duration (months)",
                min_value=0.0,
                value=float(
                    preset.get(
                        "duration_months",
                        get_default("duration_months", 12)
                    )
                )
            )

            sites = st.number_input(
                "Number of sites",
                min_value=0,
                value=int(
                    preset.get(
                        "n_sites",
                        get_default("n_sites", 10)
                    )
                )
            )

            enrollment = st.number_input(
                "Enrollment",
                min_value=0,
                value=int(
                    preset.get(
                        "enrollment",
                        get_default("enrollment", 100)
                    )
                )
            )

            year = st.number_input(
                "Start year",
                min_value=1900,
                max_value=2100,
                value=int(
                    preset.get(
                        "start_year",
                        get_default("start_year", 2024)
                    )
                )
            )

            phase_options = get_options(
                "phase",
                ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
            )

            phase = st.selectbox(
                "Phase",
                phase_options,
                index=(
                    phase_options.index(preset["phase"])
                    if preset.get("phase") in phase_options
                    else 0
                )
            )

            allocation_options = get_options(
                "allocation",
                ["Randomized", "Non-Randomized"]
            )

            allocation = st.selectbox(
                "Allocation",
                allocation_options,
                index=(
                    allocation_options.index(preset["allocation"])
                    if preset.get("allocation") in allocation_options
                    else 0
                )
            )

        with col2:

            masking_options = get_options(
                "masking",
                ["None", "Single", "Double", "Triple", "Quadruple"]
            )

            if "None" not in masking_options:
                masking_options.insert(0, "None")

            masking = st.selectbox(
                "Masking",
                masking_options,
                index=(
                    masking_options.index(preset["masking"])
                    if preset.get("masking") in masking_options
                    else 0
                )
            )

            age = st.number_input(
                "Minimum age",
                min_value=0,
                max_value=120,
                value=int(
                    preset.get(
                        "age_min",
                        get_default("age_min", 18)
                    )
                )
            )

            gender_options = get_options(
                "gender",
                ["All", "Male", "Female"]
            )

            gender = st.selectbox(
                "Gender",
                gender_options,
                index=(
                    gender_options.index(preset["gender"])
                    if preset.get("gender") in gender_options
                    else 0
                )
            )

        st.divider()

        st.subheader("Additional Trial Information")

        criteria = st.text_area(
            "Eligibility criteria",
            value=preset.get("criteria", "")
        )

        conditions = st.text_area(
            "Conditions",
            value=preset.get("conditions", "")
        )

        drugs = st.text_area(
            "Drugs / Interventions",
            value=preset.get("drugs", "")
        )

        summary = st.text_area(
            "Brief summary",
            value=preset.get("brief_summary", "")
        )

        sponsor = st.text_input(
            "Sponsor",
            value=preset.get("sponsor", "")
        )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        values = {
            "duration_months": duration,
            "n_sites": sites,
            "enrollment": enrollment,
            "start_year": year,
            "phase": phase,
            "allocation": allocation,
            "masking": masking,
            "age_min": age,
            "gender": gender,
            "criteria": criteria,
            "conditions": conditions,
            "drugs": drugs,
            "brief_summary": summary,
            "sponsor": sponsor
        }

        trial_df = pd.DataFrame(
            [values],
            columns=RAW_COLUMNS
        )

        try:

            A = builder.transform(trial_df)

            raw_prediction = float(
                model.predict(A)[0]
            )

            displayed_prediction = float(
                np.clip(raw_prediction, 0, 1)
            )

            lower, upper = make_interval(A)

            low = float(lower[0])
            high = float(upper[0])

            width = high - low

            abstain = width > width_threshold

            st.session_state["last_trial"] = values
            st.session_state["last_prediction"] = raw_prediction
            st.session_state["last_interval"] = (low, high)
            st.session_state["last_width"] = width
            st.session_state["last_abstain"] = abstain

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated Dropout Rate (clipped)",
                    percentage(displayed_prediction)
                )

            with col2:
                st.metric(
                    "Raw Model Prediction",
                    f"{raw_prediction:.4f}"
                )

            st.subheader("Conformal Prediction Interval")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Lower Bound",
                percentage(low)
            )

            col2.metric(
                "Upper Bound",
                percentage(high)
            )

            col3.metric(
                "Interval Width",
                f"{width:.4f}"
            )

            st.caption(
                f"Nominal conformal confidence level: "
                f"{(1-alpha)*100:.0f}%"
            )

            if raw_prediction < 0 or raw_prediction > 1:

                st.error(
                    "The raw prediction is outside the valid "
                    "0–1 range. The displayed value is clipped. "
                    "This prediction requires investigation."
                )

            else:

                st.success(
                    "The raw prediction is within the valid "
                    "0–1 range."
                )

            st.subheader("Reliability Assessment")

            st.metric(
                "Dashboard Abstention Threshold",
                f"{width_threshold:.4f}"
            )

            if abstain:

                st.warning(
                    "ABSTENTION: The prediction interval is "
                    "wider than the calibration-based dashboard "
                    "threshold. Additional evidence or expert "
                    "review is recommended."
                )

            else:

                st.success(
                    "RETAINED: The interval width is within "
                    "the dashboard threshold."
                )

            st.caption(
                "This threshold is the 90th percentile of "
                "calibration interval widths. It is a dashboard "
                "decision rule, not a clinical safety threshold."
            )

            st.divider()

            # INDIVIDUAL SHAP

            st.subheader("Individual Prediction Explanation")

            shap_values, base_value, feature_names = (
                get_shap_explanation(A)
            )

            if len(shap_values) == len(feature_names):

                explanation_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Contribution": shap_values
                })

                explanation_df["Absolute Contribution"] = (
                    explanation_df["Contribution"].abs()
                )

                explanation_df = explanation_df.sort_values(
                    "Absolute Contribution",
                    ascending=False
                )

                top = explanation_df.head(12)

                plot_data = top.sort_values(
                    "Contribution"
                )

                fig, ax = plt.subplots(figsize=(10, 6))

                ax.barh(
                    plot_data["Feature"],
                    plot_data["Contribution"]
                )

                ax.axvline(
                    0,
                    color="black",
                    linewidth=1
                )

                ax.set_xlabel(
                    "Contribution to raw model prediction"
                )

                ax.set_title(
                    "Individual Feature Contributions"
                )

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.info(
                    f"Baseline prediction: {base_value:.4f}. "
                    "Positive contributions increase the raw "
                    "model output relative to the baseline; "
                    "negative contributions decrease it."
                )

                st.caption(
                    "SHAP describes model associations, not "
                    "causal effects. Text components are learned "
                    "representations rather than individual words."
                )

                with st.expander("View all feature contributions"):

                    st.dataframe(
                        explanation_df,
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.warning(
                    "Could not align SHAP values with feature names."
                )

            # DOWNLOAD REPORT

            report = pd.DataFrame([{
                **values,
                "raw_model_prediction": raw_prediction,
                "displayed_prediction": displayed_prediction,
                "lower_bound": low,
                "upper_bound": high,
                "interval_width": width,
                "abstention_threshold": width_threshold,
                "status": (
                    "Abstained" if abstain else "Retained"
                )
            }])

            st.download_button(
                "Download Prediction Report",
                report.to_csv(index=False),
                file_name="clinical_trial_prediction.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==========================================================
# SECTION 3: EXPLAINABILITY & EVIDENCE
# ==========================================================

elif page == "Explainability & Evidence":

    st.header("Explainability & Evidence Analysis")

    st.write(
        "Explore global feature importance and evaluate "
        "how missing trial information affects uncertainty."
    )

    # GLOBAL SHAP

    st.subheader("Global Feature Importance")

    shap_image = OUT / "shap_feature_importance.png"

    if shap_image.exists():

        st.image(
            str(shap_image),
            use_container_width=True
        )

    else:

        st.info(
            "Global SHAP feature importance image not found."
        )

    shap_table = read_csv(
        "shap_feature_importance.csv"
    )

    if shap_table is not None:

        st.subheader("Most Influential Features")

        st.dataframe(
            shap_table.head(15),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Feature importance indicates contribution magnitude, "
        "not causality or direction of effect."
    )

    st.divider()

    # INDIVIDUAL EVIDENCE ANALYSIS

    st.subheader("Evidence-Guided Analysis")

    st.write(
        "Compare the original prediction interval with intervals "
        "obtained after hiding groups of trial information."
    )

    last_trial = st.session_state.get("last_trial")

    if last_trial is None:

        st.info(
            "Generate a prediction first in the Predict section "
            "to perform individual evidence analysis."
        )

    else:

        st.success(
            "A previous prediction is available for analysis."
        )

        if st.button(
            "Analyze Evidence for Last Prediction"
        ):

            trial_df = pd.DataFrame(
                [last_trial],
                columns=RAW_COLUMNS
            )

            try:

                results = evidence_analysis(trial_df)

                st.session_state["last_evidence"] = results

            except Exception as error:

                st.error(
                    f"Evidence analysis failed: {error}"
                )

        evidence = st.session_state.get(
            "last_evidence"
        )

        if evidence is not None:

            st.dataframe(
                evidence,
                use_container_width=True,
                hide_index=True
            )

            chart_data = evidence.sort_values(
                "Increase After Hiding"
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.barh(
                chart_data["Evidence Group"],
                chart_data["Increase After Hiding"]
            )

            ax.axvline(
                0,
                color="black",
                linewidth=1
            )

            ax.set_xlabel(
                "Change in prediction interval width"
            )

            ax.set_title(
                "Effect of Hiding Evidence Groups"
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            best = evidence.iloc[0]

            if best["Increase After Hiding"] > 0:

                st.success(
                    f"Largest observed increase in uncertainty: "
                    f"{best['Evidence Group']}"
                )

            else:

                st.info(
                    "Hiding the tested evidence groups did not "
                    "increase interval width for this trial."
                )

            st.caption(
                "This is a model-based masking experiment. "
                "It does not establish a causal benefit from "
                "collecting a particular type of evidence."
            )

    st.divider()

    # PREVIOUS VALIDATION RESULTS

    st.subheader("Previously Validated Evidence Results")

    validated = read_csv(
        "evidence_validation_comparison.csv"
    )

    if validated is not None:

        st.dataframe(
            validated,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Evidence validation results not found."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Clinical Trial Retention Intelligence | "
    "Academic Research Prototype"
)
