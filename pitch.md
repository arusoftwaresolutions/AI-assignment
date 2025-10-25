Slide 1 — Problem & SDG alignment
- Problem: Policymakers need reliable forecasts of CO₂ emissions to meet SDG 13.
- Impact: Accurate per-capita CO₂ estimates inform mitigation priority & resource allocation.

---
Slide 2 — AI/ML solution overview
- Supervised regression to map socio-economic & energy features to CO₂ per capita.
- Models: RandomForestRegressor (primary) + LinearRegression baseline.
- User-friendly Streamlit UI for interactive predictions.

---
Slide 3 — Dataset & methods
- Dataset: Included sample CSV (demo). Production: World Bank / IEA / EDGAR recommended.
- Preprocessing: scaling, train/test split.
- Evaluation: MAE, RMSE, R²; feature importance & regression plots.

---
Slide 4 — Results & impact
- Quick insights: feature importance shows energy use & GDP are strong drivers.
- Interactive tool allows scenario testing (e.g., increase renewables -> lower CO₂).
- Use-case: regional planners, NGO analysts for rapid prototyping.

---
Slide 5 — Ethics & future work
- Ethics: dataset bias, transparency, avoid overreliance on model without domain checks.
- Next steps: larger datasets, time-series models, uncertainty quantification, deployment to Streamlit Cloud / Render.