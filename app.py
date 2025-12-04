import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ---------------- Load Dataset ----------------
df = pd.read_csv(r"C:\Users\jayas\OneDrive\Documents\Desktop\co2_Emission_project\co2.csv.csv")
df.columns = df.columns.str.strip()




# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🚗 CO2 Emissions Project")
    page = st.selectbox("Choose Section", ["Visualization", "Model"])

# ---------------- Visualization ----------------
if page == "Visualization":
    st.title("📊 CO2 Emissions Data Visualization")

    st.subheader("Dataset Preview")
    st.write(df.head())

    # CO2 vs Engine Size
    st.subheader("CO2 Emissions vs Engine Size")
    fig1 = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Engine Size(L)", y="CO2 Emissions(g/km)", hue="Fuel Type", alpha=0.6)
    plt.title("CO2 Emissions by Engine Size and Fuel Type")
    st.pyplot(fig1)

    # CO2 vs Cylinders
    st.subheader("CO2 Emissions vs Cylinders")
    fig2 = plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Cylinders", y="CO2 Emissions(g/km)")
    plt.title("CO2 Emissions by Number of Cylinders")
    st.pyplot(fig2)

    # CO2 vs Fuel Consumption
    st.subheader("CO2 Emissions vs Fuel Consumption")
    fig3 = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Fuel Consumption Comb (L/100 km)", y="CO2 Emissions(g/km)", hue="Fuel Type", alpha=0.6)
    plt.title("CO2 Emissions by Fuel Consumption")
    st.pyplot(fig3)

    # Average CO2 by Fuel Type
    st.subheader("Average CO2 Emissions by Fuel Type")
    avg_fuel = df.groupby("Fuel Type")["CO2 Emissions(g/km)"].mean().reset_index()
    fig4 = plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_fuel, x="Fuel Type", y="CO2 Emissions(g/km)")
    plt.title("Average CO2 Emissions per Fuel Type")
    st.pyplot(fig4)

# ---------------- Model ----------------
else:
    st.title("🤖 CO2 Emission Prediction Model")

    # Define features and target
    X = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'Fuel Type']]
    y = df['CO2 Emissions(g/km)']

    # Preprocessing
    numeric_features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
    categorical_features = ['Fuel Type']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Build pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # ---------------- Streamlit Inputs ----------------
    st.write("### Enter Vehicle Specifications:")

    engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=3000.0, step=0.1)
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1)
    fuel_cons = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=1.0, max_value=100.0, step=0.1)
    fuel_type = st.selectbox("Fuel Type", df['Fuel Type'].unique())

    # Prepare input
    input_data = pd.DataFrame({
        'Engine Size(L)': [engine_size],
        'Cylinders': [cylinders],
        'Fuel Consumption Comb (L/100 km)': [fuel_cons],
        'Fuel Type': [fuel_type]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"Predicted CO2 Emissions: {prediction:.2f} g/km")
