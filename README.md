# Telco Churn Prediction

This repository contains a Telco customer churn analysis and prediction project.
It includes data exploration, preprocessing, model training (Random Forest, XGBoost, CatBoost),
explainability (SHAP & LIME), and a small Streamlit app for inference.

## Contents

- `main.ipynb`  analysis, modelling, evaluation, and explanation notebooks.
- `app.py`  Streamlit app generated from the notebook for running predictions.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`  original dataset (included).
- `churn_models_bundle.pkl`  trained models and metadata (binary, large).
- `requirements.txt`  Python dependencies.

## Quickstart

1. Create and activate a virtual environment (Windows):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

## Notes

- The repository currently contains dataset and model bundle files which are large. If you do not want
  these in the remote history, I can remove them and rewrite history (force-push). Tell me if you'd like that.
- A `.gitignore` was added to exclude common virtualenv, caches and temporary files.

## License

See the `LICENSE` file in the repository.
