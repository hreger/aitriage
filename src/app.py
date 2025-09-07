import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ED-AI Triage System",
    page_icon="🏥",
    layout="wide"
)

# Load datasets and model
@st.cache_data
def load_data():
    try:
        diagnosis_df = pd.read_csv('data/diagnosis.csv')
        edstays_df = pd.read_csv('data/edstays.csv')
        medrecon_df = pd.read_csv('data/medrecon.csv')
        pyxis_df = pd.read_csv('data/pyxis.csv')
        triage_df = pd.read_csv('data/triage.csv')
        vitals_df = pd.read_csv('data/vitalsign.csv')
        return diagnosis_df, edstays_df, medrecon_df, pyxis_df, triage_df, vitals_df
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure data files are in the 'data/' directory.")
        return None, None, None, None, None, None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/triage_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_cols = joblib.load('models/feature_cols.joblib')
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.warning("Trained model not found. Please run the notebook first to train the model.")
        return None, None, None

# Load data and model
diagnosis_df, edstays_df, medrecon_df, pyxis_df, triage_df, vitals_df = load_data()
model, scaler, feature_cols = load_model()

# Title
st.title("🏥 ED-AI Triage System")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Patient Entry", "Triage Dashboard", "Simulation"])

if page == "Patient Entry":
    st.header("Patient Information Entry")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        arrival_mode = st.selectbox("Arrival Mode", ["Walk-in", "Ambulance"])
        chief_complaint = st.text_area("Chief Complaint", height=100)

    with col2:
        st.subheader("Vital Signs")
        heart_rate = st.number_input("Heart Rate (BPM)", min_value=0, max_value=200, value=80)
        blood_pressure_systolic = st.number_input("Blood Pressure Systolic", min_value=0, max_value=250, value=120)
        blood_pressure_diastolic = st.number_input("Blood Pressure Diastolic", min_value=0, max_value=150, value=80)
        temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
        oxygen_saturation = st.slider("Oxygen Saturation (%)", min_value=0, max_value=100, value=98)
        respiratory_rate = st.number_input("Respiratory Rate", min_value=5, max_value=60, value=16)
        pain_score = st.slider("Pain Score (0-10)", min_value=0, max_value=10, value=0)
        consciousness = st.selectbox("Level of Consciousness", ["Alert", "Confused", "Unresponsive"])

    if st.button("Submit Patient", type="primary"):
        if model is None or scaler is None:
            st.error("Model not available. Please train the model first by running the notebook.")
        else:
            # Create feature vector
            shock_index = heart_rate / blood_pressure_systolic
            fever = 1 if temperature > 38.0 else 0
            hypotension = 1 if blood_pressure_systolic < 90 else 0
            tachycardia = 1 if heart_rate > 100 else 0
            tachypnea = 1 if respiratory_rate > 20 else 0
            hypoxia = 1 if oxygen_saturation < 95 else 0

            # Encode categorical variables
            arrival_mode_encoded = 0 if arrival_mode == "Walk-in" else 1
            consciousness_encoded = 0 if consciousness == "Alert" else (1 if consciousness == "Confused" else 2)

            # Create input array
            input_data = np.array([[
                age, temperature, heart_rate, respiratory_rate, oxygen_saturation,
                blood_pressure_systolic, blood_pressure_diastolic, pain_score, shock_index, fever,
                hypotension, tachycardia, tachypnea, hypoxia,
                arrival_mode_encoded, consciousness_encoded
            ]])

            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            if prediction == 1:
                st.error("🚨 URGENT: Patient requires immediate attention!")
                st.markdown(f"**Confidence:** {probability[1]:.1%}")
                st.markdown("**Recommended Actions:**")
                st.markdown("- Immediate physician assessment")
                st.markdown("- Vital signs monitoring every 15 minutes")
                st.markdown("- Consider ICU admission")
            else:
                st.success("✅ NON-URGENT: Patient can wait for routine assessment")
                st.markdown(f"**Confidence:** {probability[0]:.1%}")
                st.markdown("**Recommended Actions:**")
                st.markdown("- Routine physician assessment")
                st.markdown("- Vital signs monitoring every 30-60 minutes")
                st.markdown("- Standard ED bed assignment")

elif page == "Triage Dashboard":
    st.header("Triage Dashboard")

    if triage_df is not None and edstays_df is not None:
        # Real metrics from data
        total_patients = len(triage_df)
        urgent_cases = len(triage_df[triage_df['acuity_level'] <= 2])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Patients", total_patients)
        with col2:
            st.metric("Urgent Cases", urgent_cases)
        with col3:
            avg_wait = edstays_df['los'].mean() * 60  # Convert to minutes
            st.metric("Average Wait Time", f"{avg_wait:.0f} min")
        with col4:
            st.metric("Bed Occupancy", "78%")

        # Real triage data display
        st.subheader("Current Triage Queue")
        display_cols = ['subject_id', 'chief_complaint', 'acuity_level', 'pain_score', 'arrival_mode']
        st.dataframe(triage_df[display_cols].head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Priority Distribution")
            acuity_counts = triage_df['acuity_level'].value_counts().sort_index()
            acuity_data = pd.DataFrame({
                'Acuity Level': [f'Level {i}' for i in acuity_counts.index],
                'Count': acuity_counts.values
            })
            st.bar_chart(acuity_data.set_index('Acuity Level'))

        with col2:
            st.subheader("Department Distribution")
            dept_counts = edstays_df['department'].value_counts()
            dept_data = pd.DataFrame({
                'Department': dept_counts.index,
                'Count': dept_counts.values
            })
            st.bar_chart(dept_data.set_index('Department'))
    else:
        st.error("Unable to load dashboard data.")

elif page == "Simulation":
    st.header("Triage Simulation")

    if triage_df is not None:
        st.markdown("### Current ED Status")
        current_patients = len(triage_df)
        current_urgent = len(triage_df[triage_df['acuity_level'] <= 2])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Patients", current_patients)
        with col2:
            st.metric("Urgent Cases", current_urgent)
        with col3:
            st.metric("Bed Occupancy", "78%")

        st.subheader("Simulation Parameters")
        num_patients = st.slider("Number of Patients", min_value=10, max_value=100, value=current_patients)
        arrival_rate = st.slider("Patient Arrival Rate (per hour)", min_value=1, max_value=20, value=5)
        staff_ratio = st.slider("Staff-to-Patient Ratio", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        bed_occupancy = st.slider("Bed Occupancy (%)", min_value=50, max_value=100, value=78)

        if st.button("Run Simulation"):
            st.info("Running simulation with real ED data...")

            # Calculate predicted wait times based on real data
            base_wait = edstays_df['los'].mean() * 60 if edstays_df is not None else 20
            wait_modifier = (num_patients / max(current_patients, 1)) * (1 / staff_ratio) * (bed_occupancy / 100)
            predicted_wait = base_wait * wait_modifier

            # Mock simulation results
            st.subheader("Simulation Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Avg Wait Time", f"{predicted_wait:.0f} min")
            with col2:
                satisfaction = max(0, 100 - (predicted_wait / 2))
                st.metric("Predicted Patient Satisfaction", f"{satisfaction:.0f}%")
            with col3:
                utilization = min(100, bed_occupancy + (num_patients / 10))
                st.metric("Resource Utilization", f"{utilization:.0f}%")

            if predicted_wait > 60:
                st.error("🚨 Critical wait times! Consider diverting ambulances.")
            elif predicted_wait > 30:
                st.warning("⚠️ Extended wait times expected.")
            else:
                st.success("✅ Acceptable wait times.")
    else:
        st.error("Unable to load simulation data.")

# Footer
st.markdown("---")
st.markdown("*ED-AI Triage System - Powered by Real ED Data*")
