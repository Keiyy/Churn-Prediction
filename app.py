
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn")

@st.cache_data
def load_bundle(path="churn_models_bundle.pkl"):
    bundle = joblib.load(path)
    return bundle

try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Failed to load churn_models_bundle.pkl: {e}")
    st.stop()

models_map = {
    "Random Forest": bundle["models"]["rf"],
    "XGBoost": bundle["models"]["xgb"],
    "CatBoost": bundle["models"]["cat"],
}

thresholds = bundle.get("thresholds", {})
enc_cols = bundle["feature_columns"]["encoded"]
raw_cols = bundle["feature_columns"]["raw"]

def preprocess_raw_to_encoded(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # robust mapping for booleans and simple categories
    def map_bool_val(v):
        if pd.isna(v):
            return v
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        s = str(v).strip().lower()
        if s in ("yes", "y", "true", "t", "1"):
            return 1
        if s in ("no", "n", "false", "f", "0"):
            return 0
        if s == "male":
            return 1
        if s == "female":
            return 0
        return v

    for col in [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
        "SeniorCitizen",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(map_bool_val)

    if "Contract" in df.columns:
        contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["Contract"] = df["Contract"].map(contract_mapping).fillna(df["Contract"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df_d = pd.get_dummies(df, drop_first=True)
    df_enc = df_d.reindex(columns=enc_cols, fill_value=0)
    return df_enc

def predict_with_bundle(model_name: str, X_enc: pd.DataFrame = None, X_raw: pd.DataFrame = None):
    thr = thresholds.get(model_name, 0.5)
    model = models_map[model_name]

    if model_name in ["Random Forest", "XGBoost"]:
        if X_enc is None:
            raise ValueError("Encoded DataFrame required for RF/XGB")
        X_enc = X_enc.reindex(columns=enc_cols, fill_value=0)
        proba = model.predict_proba(X_enc)[:, 1]
    else:
        if X_raw is None:
            raise ValueError("Raw DataFrame required for CatBoost")
        X_raw = X_raw.reindex(columns=raw_cols)
        proba = model.predict_proba(X_raw)[:, 1]

    preds = (proba >= thr).astype(int)
    return proba, preds

st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Choose model", list(models_map.keys()))
mode = st.sidebar.radio("Input mode", ["Single customer", "Batch CSV upload"])
show_expl = st.sidebar.checkbox("Enable explanations (SHAP / LIME)")
expl_method = st.sidebar.selectbox("Method", ["SHAP", "LIME"]) if show_expl else None

if mode == "Single customer":
    st.header("Single customer features")
    with st.form(key="single_form"):
        inputs = {}
        col1, col2 = st.columns(2)
        for i, c in enumerate(raw_cols):
            widget_col = col1 if i % 2 == 0 else col2
            if any(k in c.lower() for k in ["tenure", "monthlycharges", "totalcharges"]):
                val = widget_col.number_input(c, value=0.0, format="%.2f")
                inputs[c] = val
            else:
                if c == "gender":
                    inputs[c] = widget_col.selectbox(c, ["Female", "Male"])
                elif c == "MultipleLines":
                    inputs[c] = widget_col.selectbox(c, ["Yes", "No", "No phone service"])
                elif c == "OnlineSecurity":
                    inputs[c] = widget_col.selectbox(c, ["Yes", "No", "No internet service"])
                elif c in [
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "PaperlessBilling",
                    "SeniorCitizen",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ]:
                    # restrict boolean/service flags to True/False choices
                    inputs[c] = widget_col.selectbox(c, [True, False])
                elif c == "Contract":
                    inputs[c] = widget_col.selectbox(c, ["Month-to-month", "One year", "Two year"])
                elif c == "InternetService":
                    inputs[c] = widget_col.selectbox(c, ["DSL", "Fiber optic", "No"])
                elif c == "PaymentMethod":
                    inputs[c] = widget_col.selectbox(c, [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ])
                else:
                    inputs[c] = widget_col.text_input(c, value="")

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([inputs], columns=raw_cols)
        try:
            if model_choice in ["Random Forest", "XGBoost"]:
                X_enc = preprocess_raw_to_encoded(row)
                proba, pred = predict_with_bundle(model_choice, X_enc=X_enc)
            else:
                proba, pred = predict_with_bundle(model_choice, X_raw=row)

            st.success(f"Predicted probability of churn: {proba[0]:.4f}")
            thr = thresholds.get(model_choice, 0.5)
            st.info(f"Threshold: {thr} → Predicted class: {int(pred[0])} (1 = churn)")
            # Explanations (single instance)
            if show_expl:
                st.markdown("**Explanations**")

                # prepare background data
                try:
                    df_full = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
                    bg_df = df_full
                except Exception:
                    bg_df = row.copy()

                # SHAP is disabled; prefer LIME for now
                if expl_method == "SHAP":
                    st.info("SHAP explanations are temporarily disabled (use LIME).")

                try:
                    if expl_method == "LIME" or expl_method is None:
                        # Encoded models (Random Forest / XGBoost)
                        if model_choice in ["Random Forest", "XGBoost"]:
                            # build background encoded matrix
                            Xtr = bg_df[raw_cols].copy() if set(raw_cols).issubset(bg_df.columns) else bg_df.copy()
                            Xtr_enc = preprocess_raw_to_encoded(Xtr[raw_cols]) if set(raw_cols).issubset(Xtr.columns) else preprocess_raw_to_encoded(Xtr)
                            feature_names = Xtr_enc.columns.tolist()
                            Xtr_lime = Xtr_enc.astype(float)
                            Xte_lime = preprocess_raw_to_encoded(row)

                            def predict_fn(X_array):
                                X_df = pd.DataFrame(X_array, columns=feature_names)
                                return models_map[model_choice].predict_proba(X_df)

                            explainer = LimeTabularExplainer(
                                training_data=Xtr_lime.values,
                                feature_names=feature_names,
                                class_names=["No Churn", "Churn"],
                                mode="classification",
                                discretize_continuous=True,
                            )

                            exp = explainer.explain_instance(Xte_lime.iloc[0].values, predict_fn, num_features=10)
                            fig = exp.as_pyplot_figure()
                            st.pyplot(fig)
                            st.write("Top features:")
                            for feat, weight in exp.as_list():
                                st.write(f"{feat}: {weight:+.4f}")

                        else:
                            # CatBoost (raw categorical) — encode numerics and map categories for LIME
                            Xtr = bg_df[raw_cols].copy()
                            cat_cols = Xtr.select_dtypes(include=['object']).columns.tolist()
                            num_cols = [c for c in Xtr.columns if c not in cat_cols]
                            cat_maps = {}
                            for c in cat_cols:
                                cats = pd.Series(Xtr[c].astype(str).fillna("MISSING")).unique().tolist()
                                cat_maps[c] = cats

                            def encode_df_for_lime(df):
                                df2 = df.copy()
                                for c in num_cols:
                                    df2[c] = pd.to_numeric(df2[c], errors='coerce').fillna(0.0).astype(float)
                                for c in cat_cols:
                                    cats = cat_maps[c]
                                    s = df2[c].astype(str).fillna("MISSING")
                                    s = s.where(s.isin(cats), other="UNKNOWN")
                                    if "UNKNOWN" not in cats:
                                        cat_maps[c].append("UNKNOWN")
                                        cats = cat_maps[c]
                                    df2[c] = s.map({v: k for k, v in enumerate(cats)}).astype(int)
                                return df2

                            Xtr_lime = encode_df_for_lime(Xtr)
                            Xte_lime = encode_df_for_lime(row[raw_cols])
                            feature_names = Xtr_lime.columns.tolist()

                            def predict_fn_cat(X_array):
                                X_df = pd.DataFrame(X_array, columns=feature_names)
                                for c in num_cols:
                                    X_df[c] = pd.to_numeric(X_df[c], errors='coerce').fillna(0.0)
                                for c in cat_cols:
                                    cats = cat_maps[c]
                                    X_df[c] = X_df[c].round().astype(int).clip(0, len(cats)-1).map(lambda k: cats[k])
                                return models_map[model_choice].predict_proba(X_df)

                            explainer = LimeTabularExplainer(
                                training_data=Xtr_lime.values,
                                feature_names=feature_names,
                                class_names=["No Churn", "Churn"],
                                mode="classification",
                                discretize_continuous=True,
                                categorical_features=[feature_names.index(c) for c in cat_cols],
                                categorical_names={feature_names.index(c): cat_maps[c] for c in cat_cols},
                            )

                            exp = explainer.explain_instance(Xte_lime.iloc[0].values, predict_fn_cat, num_features=10)
                            fig = exp.as_pyplot_figure()
                            st.pyplot(fig)
                            st.write("Top features:")
                            for feat, weight in exp.as_list():
                                st.write(f"{feat}: {weight:+.4f}")

                except Exception as e:
                    st.error(f"Explanation failed: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.header("Batch CSV upload")
    st.write("Upload a CSV containing either raw columns (original dataset columns) or encoded columns used by the RF/XGB models.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df_in.head())

        run_btn = st.button("Run predictions on uploaded file")
        if run_btn:
            try:
                if model_choice in ["Random Forest", "XGBoost"]:
                    if set(raw_cols).issubset(df_in.columns):
                        X_enc = preprocess_raw_to_encoded(df_in[raw_cols])
                    else:
                        X_enc = df_in.reindex(columns=enc_cols, fill_value=0)

                    proba, pred = predict_with_bundle(model_choice, X_enc=X_enc)
                    out = df_in.copy()
                    out["churn_proba"] = proba
                    out["churn_pred"] = pred

                else:
                    if set(raw_cols).issubset(df_in.columns):
                        X_raw = df_in[raw_cols]
                    else:
                        st.warning("Uploaded CSV doesn't contain required raw columns for CatBoost. Please upload raw-format CSV.")
                        st.stop()

                    proba, pred = predict_with_bundle(model_choice, X_raw=X_raw)
                    out = df_in.copy()
                    out["churn_proba"] = proba
                    out["churn_pred"] = pred

                st.success("Predictions complete — preview below")
                st.dataframe(out.head())

                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# st.sidebar.markdown("---")
# st.sidebar.write("Bundle versions:")
# for k, v in bundle.get("versions", {}).items():
#     st.sidebar.write(f"{k}: {v}")
