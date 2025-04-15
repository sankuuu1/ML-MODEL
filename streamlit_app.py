
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Load Data
stress_strain_df = pd.read_excel("Stress_Strain_Pattern_Thickness.xlsx", sheet_name="Sheet1")
uts_elongation_df = pd.read_excel("UTS_Elongation.xlsx", skiprows=2, usecols="B:E")
uts_elongation_df.columns = ['Layer_Thickness_mm', 'Pattern', 'UTS_MPa', 'Elongation_percent']
uts_elongation_df.dropna(inplace=True)

# Prepare Data
patterns = ['Cubic', 'Gyroid', 'Hexagonal']
num_columns = stress_strain_df.shape[1]
num_thickness_groups = num_columns // 6
layer_thicknesses = [0.12 + 0.08 * i for i in range(num_thickness_groups)]

records = []
for i in range(num_thickness_groups):
    col_start = i * 6
    thickness = layer_thicknesses[i]
    for j, pattern in enumerate(patterns):
        stress_col = stress_strain_df.columns[col_start + j * 2]
        strain_col = stress_strain_df.columns[col_start + j * 2 + 1]
        data = stress_strain_df[[stress_col, strain_col]].copy()
        data.columns = ['Stress', 'Strain']
        data.dropna(inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        if not data.empty:
            uts = data['Stress'].max()
            elong = data['Strain'].max()
            records.append({
                'Layer_Thickness_mm': thickness,
                'Pattern': pattern,
                'UTS_MPa': uts,
                'Elongation_percent': elong * 100
            })

df = pd.concat([uts_elongation_df, pd.DataFrame(records)], ignore_index=True).dropna()

X = df[['Layer_Thickness_mm', 'Pattern']]
y = df[['UTS_MPa', 'Elongation_percent']]

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(), ['Pattern']),
    ('scale', StandardScaler(), ['Layer_Thickness_mm'])
])

model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

model.fit(X, y)

# Streamlit UI
st.title("Stress-Strain UTS & Elongation Predictor")

thickness = st.number_input("Enter Layer Thickness (mm)", min_value=0.1, max_value=1.0, step=0.01)
pattern = st.selectbox("Select Pattern", ["Cubic", "Gyroid", "Hexagonal"])

if st.button("Predict"):
    input_df = pd.DataFrame([{'Layer_Thickness_mm': thickness, 'Pattern': pattern}])
    prediction = model.predict(input_df)[0]
    uts, elongation = prediction

    st.success(f"Predicted UTS: {uts:.2f} MPa")
    st.success(f"Predicted Elongation: {elongation:.2f} %")

    # Plot stress-strain curve
    strain = np.linspace(0, elongation / 100, 500)
    stress = np.piecewise(
        strain,
        [strain < 0.02, (strain >= 0.02) & (strain < 0.06), strain >= 0.06],
        [
            lambda x: (uts / 0.02) * x,
            lambda x: -200 * (x - 0.02)**2 + uts,
            lambda x: -300 * (x - 0.06) + uts * 0.9
        ]
    )
    stress = np.maximum(stress, 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(strain * 100, stress, label="Stress-Strain", color='royalblue')
    ax.axvline(elongation, color='red', linestyle='--', label='Elongation')
    ax.axhline(uts, color='green', linestyle='--', label='UTS')
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Stress-Strain Curve for {pattern} ({thickness} mm)")
    ax.legend()
    st.pyplot(fig)
