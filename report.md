SDG13 — CO₂ Emissions Forecast (1-Page Summary)

SDG problem addressed
- UN SDG 13 (Climate Action): forecasting per-capita CO₂ emissions supports planning & policy by estimating emissions given socio-economic and energy indicators.

ML method used
- Supervised regression. Primary model: RandomForestRegressor; baseline: LinearRegression.
- Features: GDP per capita, population (millions), energy use per capita (GJ), renewable energy percentage, year.
- Dataset: small included sample (data/sample_co2.csv). Intended as demonstration; replace with World Bank / EDGAR data for production.

Key results (on sample split)
- Random Forest MAE: ~ (displayed in app)
- RMSE and R² also shown in app.
- Visual outputs: feature importance, Actual vs Predicted scatter.

Ethical reflection
- Risk of bias from limited dataset and omitted variables (policy, sector mix).
- Predictions should not be used for unilateral policy decisions without domain validation.
- Recommend transparency (model cards), regular re-evaluation, and inclusion of marginalized regions' data to reduce bias.

Future scope
- Integrate larger global datasets (World Bank, IEA, EDGAR).
- Add temporal models (ARIMA, LSTM) for time-series forecasting.
- Provide uncertainty estimates and counterfactual analysis for policy simulations.