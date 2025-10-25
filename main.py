import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(page_title="SDG13 — CO₂ Emissions Forecast MVP", layout="wide")

DATA_PATH = os.path.join("data", "sample_co2.csv")

@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

@st.cache_data
def preprocess(df):
    df = df.copy()
    # Drop country + year for model features; keep year if desired
    df['population_millions'] = df['population_millions'].astype(float)
    features = ['gdp_per_capita', 'population_millions', 'energy_use_per_capita', 'renewable_pct', 'year']
    X = df[features]
    y = df['co2_per_capita']
    return X, y, df

@st.cache_resource
def train_models(X_train, y_train):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    lr = LinearRegression()
    rf.fit(Xt, y_train)
    lr.fit(Xt, y_train)
    return rf, lr, scaler

def evaluate_model(model, scaler, X_test, y_test):
    Xt = scaler.transform(X_test)
    preds = model.predict(Xt)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    return preds, mae, rmse, r2

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    fi = pd.Series(importances, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots()
    fi.plot.barh(ax=ax, color='C0')
    ax.set_title("Feature Importance (Random Forest)")
    return fig

def plot_regression(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual CO₂ per capita (tons)")
    ax.set_ylabel("Predicted CO₂ per capita (tons)")
    ax.set_title("Actual vs Predicted")
    return fig

# UI
st.title("SDG 13 — CO₂ Emissions Forecast MVP")
st.markdown("Predict per-capita CO₂ emissions using socio-economic & energy features.\nModel: RandomForestRegressor (with LinearRegression baseline)")

df = load_data()
X, y, df_full = preprocess(df)

st.sidebar.header("Quick data preview")
st.sidebar.write(df_full.head())

st.header("1) Dataset overview")
st.dataframe(df_full)

st.markdown("### Feature distributions")
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3))
sns.histplot(df_full['co2_per_capita'], kde=True, ax=ax1)
ax1.set_title("CO₂ per capita distribution")
st.pyplot(fig1)

# Train/test split
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

with st.spinner("Training models..."):
    rf, lr, scaler = train_models(X_train, y_train)

# Evaluate
rf_preds, rf_mae, rf_rmse, rf_r2 = evaluate_model(rf, scaler, X_test, y_test)
lr_preds, lr_mae, lr_rmse, lr_r2 = evaluate_model(lr, scaler, X_test, y_test)

st.header("2) Model evaluation")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Random Forest (primary)")
    st.write(f"MAE: {rf_mae:.3f} tons")
    st.write(f"RMSE: {rf_rmse:.3f} tons")
    st.write(f"R²: {rf_r2:.3f}")
    st.pyplot(plot_regression(y_test, rf_preds))
    st.pyplot(plot_feature_importance(rf, X.columns))

with col2:
    st.subheader("Linear Regression (baseline)")
    st.write(f"MAE: {lr_mae:.3f} tons")
    st.write(f"RMSE: {lr_rmse:.3f} tons")
    st.write(f"R²: {lr_r2:.3f}")
    fig_lr, ax = plt.subplots()
    sns.regplot(x=y_test, y=lr_preds, ax=ax, scatter_kws={'s':50})
    ax.set_xlabel("Actual CO₂ per capita (tons)")
    ax.set_ylabel("Predicted CO₂ per capita (tons)")
    st.pyplot(fig_lr)

st.header("3) Interactive prediction")
st.markdown("Enter feature values to get a CO₂ per capita prediction:")

with st.form("predict_form"):
    year = st.number_input("Year", value=2024, format="%d")
    gdp = st.number_input("GDP per capita (USD)", value=8000.0)
    pop = st.number_input("Population (millions)", value=10.0, step=0.1)
    energy = st.number_input("Energy use per capita (GJ)", value=25.0)
    renewable = st.slider("Renewable energy (%)", 0.0, 100.0, 20.0)
    submitted = st.form_submit_button("Predict CO₂")

if submitted:
    X_input = pd.DataFrame([{
        'gdp_per_capita': gdp,
        'population_millions': pop,
        'energy_use_per_capita': energy,
        'renewable_pct': renewable,
        'year': year
    }])
    Xt = scaler.transform(X_input)
    pred = rf.predict(Xt)[0]
    st.success(f"Predicted CO₂ emissions per capita: {pred:.3f} tons")
    st.info("Interpretation: higher energy use and GDP tend to increase CO₂ per capita; higher renewable_pct reduces it (model-dependent).")

st.header("4) Export / Notes")
if st.button("Download trained RandomForest model"):
    joblib.dump({'model': rf, 'scaler': scaler, 'features': list(X.columns)}, "rf_model.joblib")
    with open("rf_model.joblib", "rb") as f:
        st.download_button("Download rf_model.joblib", f, file_name="rf_model.joblib")

st.markdown("Project artifacts: README.md, report.md, pitch_deck.md included in repo.")