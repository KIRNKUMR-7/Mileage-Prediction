import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
file_path = r"C:\Users\KIRAN\Downloads\air quality samp\Air_Quality.csv"
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}")
        return None

df = load_data()

if df is not None:

    st.title("Mileage Prediction App")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Mileage Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['mileage_kmpl'], kde=True, color='skyblue', edgecolor='black', ax=ax1)
    ax1.set_title("Distribution of Mileage (kmpl)")
    ax1.set_xlabel("Mileage (km/l)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    numerical_only = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numerical_only.corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    if 'mileage_kmpl' not in df.columns:
        st.error("Column 'mileage_kmpl' not found in dataset!")
        st.stop()

    X = df.drop("mileage_kmpl", axis=1)
    y = df["mileage_kmpl"]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance on Test Data")
    st.write(f"**Root Mean Squared Error (RMSE)**: `{rmse:.2f}`")
    st.write(f"**R² Score**: `{r2:.2f}`")

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, color="teal", ax=ax3)
    ax3.set_xlabel("Actual Mileage")
    ax3.set_ylabel("Predicted Mileage")
    ax3.set_title("Actual vs Predicted Mileage")
    ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    st.pyplot(fig3)

    st.subheader("Predict Mileage for a Custom Input")

    user_input = {}
    with st.form("input_form"):
        for col in numerical_cols:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        for col in categorical_cols:
            user_input[col] = st.selectbox(f"{col}", options=df[col].dropna().unique())

        submitted = st.form_submit_button("Predict Mileage")

    if submitted:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"Predicted Mileage: {prediction[0]:.2f} km/l")
