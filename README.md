# SDG13 — CO₂ Emissions Forecast MVP

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arusoftwaresolutions/AI-assignment/blob/main/app.py)

## Project Overview
**Goal:** Machine Learning MVP for **UN SDG 13 — Climate Action**

**Task:** Predict per-capita CO₂ emissions using socio-economic and energy-related features.

**Approach:** Supervised regression with **RandomForestRegressor** (primary) and **LinearRegression** (baseline).

**UI:** Streamlit app (`app.py`) that trains on included sample data and provides interactive predictions.

---

## Run Locally (Windows)
1. Open terminal in the repository folder:
```bash
cd c:\Users\admin\wk-2-assignment
```
2. Create virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
3. Run Streamlit:
```bash
streamlit run app.py
```
> Or run directly (non-interactive):
```bash
python app.py
```

---

## Repository Files
- `app.py` — Streamlit application (training, evaluation, interactive prediction)  
- `data/sample_co2.csv` — example dataset (included)  
- `requirements.txt` — dependencies  
- `report.md` — 1-page summary  
- `pitch_deck.md` — 5-slide pitch deck (Markdown)

---

## Ethics & Notes
- Small sample data for MVP; **not production-grade**.  
- Model trained on limited synthetic-like data; **real deployment requires diverse, validated datasets**.  
- No hard-coded credentials or external downloads.  
- Predictions are for **decision-support only**, not authoritative policy-making.

---

## Next Steps
- Integrate larger datasets (World Bank, UN, Kaggle)  
- Add cross-validation and hyperparameter tuning  
- Enhance interpretability (SHAP, feature importance plots)  
- Deploy interactive Streamlit app on **Render** or **Hugging Face Spaces**
