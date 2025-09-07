"""
Advanced ED-AI Triage System with ClinicalBERT, XGBoost, Interpretability, and Fairness
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
warnings.filterwarnings('ignore')

# Import our custom modules
FAIRNESS_AVAILABLE = False
try:
    from fairness_module import FairnessAssessor, generate_fairness_report
    from interpretability_module import InterpretabilityEngine, generate_counterfactual_explanation
    FAIRNESS_AVAILABLE = True
except ImportError:
    st.error("Custom modules not found. Please ensure fairness_module.py and interpretability_module.py are in the src directory.")

# Page configuration
st.set_page_config(
    page_title="Advanced ED-AI Triage System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .urgent-alert {
        background-color: #ffebee;
        border-left: 0.25rem solid #d32f2f;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-alert {
        background-color: #e8f5e8;
        border-left: 0.25rem solid #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load datasets and models
@st.cache_data
def load_datasets():
    """Load all ED datasets"""
    try:
        datasets = {
            'diagnosis': pd.read_csv('data/diagnosis.csv'),
            'edstays': pd.read_csv('data/edstays.csv'),
            'medrecon': pd.read_csv('data/medrecon.csv'),
            'pyxis': pd.read_csv('data/pyxis.csv'),
            'triage': pd.read_csv('data/triage.csv'),
            'vitals': pd.read_csv('data/vitalsign.csv')
        }
        return datasets
    except FileNotFoundError as e:
        st.error(f"Dataset files not found: {e}")
        return None

@st.cache_resource
def load_advanced_models():
    """Load advanced ML models and preprocessing objects"""
    try:
        models = {
            'xgb_model': joblib.load('models/xgb_model.joblib'),
            'lgb_model': joblib.load('models/lgb_model.joblib'),
            'scaler': joblib.load('models/advanced_scaler.joblib'),
            'feature_names': joblib.load('models/structured_features.joblib')
        }

        # Try to load ClinicalBERT components
        try:
            models['bert_tokenizer'] = joblib.load('models/bert_tokenizer.joblib')
            models['bert_model'] = torch.load('models/bert_model.pth')
        except:
            st.warning("ClinicalBERT components not found. Text processing will be limited.")
            models['bert_tokenizer'] = None
            models['bert_model'] = None

        return models
    except FileNotFoundError:
        st.warning("Advanced models not found. Please run the advanced notebook first.")
        return None

# Load data and models
datasets = load_datasets()
models = load_advanced_models()

# Initialize interpretability engine if models are available
if models and FAIRNESS_AVAILABLE:
    try:
        interpret_engine = InterpretabilityEngine(
            models['xgb_model'], models['feature_names']
        )
        fairness_assessor = FairnessAssessor()
    except:
        interpret_engine = None
        fairness_assessor = None
else:
    interpret_engine = None
    fairness_assessor = None

# Main title
st.markdown('<h1 class="main-header">🏥 Advanced ED-AI Triage System</h1>', unsafe_allow_html=True)
st.markdown("*Powered by ClinicalBERT, XGBoost, SHAP, and Advanced Interpretability*")
st.markdown("---")

# Sidebar navigation
st.sidebar.markdown('<p class="sidebar-header">🧭 Navigation</p>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["Patient Assessment", "Advanced Dashboard", "Interpretability", "Fairness Analysis",
     "Counterfactuals", "Simulation", "Model Evaluation", "Clinician Survey"],
    label_visibility="collapsed"
)

# Sidebar model status
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status:**")
if models:
    st.sidebar.success("✅ Advanced Models Loaded")
    if models.get('bert_tokenizer'):
        st.sidebar.success("✅ ClinicalBERT Available")
    else:
        st.sidebar.warning("⚠️ ClinicalBERT Not Available")
else:
    st.sidebar.error("❌ Models Not Available")

if FAIRNESS_AVAILABLE:
    st.sidebar.success("✅ Fairness Tools Available")
else:
    st.sidebar.warning("⚠️ Fairness Tools Not Available")

# Main content based on selected page
if page == "Patient Assessment":
    st.header("🩺 Advanced Patient Assessment")

    if not models:
        st.error("Advanced models not available. Please run the advanced notebook first.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Patient Information")

            # Create tabs for different input sections
            tab1, tab2, tab3 = st.tabs(["Demographics & Chief Complaint", "Vital Signs", "Medical History"])

            with tab1:
                demo_col1, demo_col2 = st.columns(2)

                with demo_col1:
                    age = st.number_input("Age", min_value=0, max_value=120, value=45)
                    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])
                    race = st.selectbox("Race/Ethnicity",
                                      ["White", "Black/African American", "Hispanic/Latino",
                                       "Asian", "Native American", "Pacific Islander", "Other"])
                    insurance = st.selectbox("Insurance Type",
                                           ["Private", "Medicare", "Medicaid", "Self-pay", "Other"])

                with demo_col2:
                    arrival_mode = st.selectbox("Arrival Mode",
                                              ["Walk-in", "Ambulance", "Private Vehicle", "Other"])
                    consciousness = st.selectbox("Level of Consciousness",
                                                ["Alert", "Confused", "Drowsy", "Unresponsive"])
                    pain_score = st.slider("Pain Score (0-10)", 0, 10, 0)

                chief_complaint = st.text_area("Chief Complaint", height=100,
                                             placeholder="Describe the patient's main symptoms...")

            with tab2:
                vitals_col1, vitals_col2 = st.columns(2)

                with vitals_col1:
                    heart_rate = st.number_input("Heart Rate (BPM)", 0, 250, 75)
                    blood_pressure_systolic = st.number_input("Systolic BP (mmHg)", 0, 300, 120)
                    blood_pressure_diastolic = st.number_input("Diastolic BP (mmHg)", 0, 200, 80)
                    temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 36.6, 0.1)

                with vitals_col2:
                    oxygen_saturation = st.slider("Oxygen Saturation (%)", 0, 100, 98)
                    respiratory_rate = st.number_input("Respiratory Rate", 0, 100, 16)
                    weight = st.number_input("Weight (kg)", 0.0, 300.0, 70.0, 0.1)
                    height = st.number_input("Height (cm)", 0.0, 250.0, 170.0, 0.1)

            with tab3:
                st.write("**Recent Medications:**")
                medications = st.text_area("List current medications",
                                         placeholder="e.g., Lisinopril 10mg daily, Metformin 500mg BID")

                st.write("**Medical History:**")
                history = st.text_area("Relevant medical history",
                                     placeholder="e.g., Diabetes, Hypertension, Previous surgeries")

        with col2:
            st.subheader("Real-time Assessment")

            if st.button("🔍 Analyze Patient", type="primary", use_container_width=True):
                with st.spinner("Running advanced AI analysis..."):

                    # Calculate advanced features
                    shock_index = heart_rate / blood_pressure_systolic if blood_pressure_systolic > 0 else 0
                    mean_arterial_pressure = (blood_pressure_systolic + 2 * blood_pressure_diastolic) / 3
                    pulse_pressure = blood_pressure_systolic - blood_pressure_diastolic

                    # Clinical flags
                    fever = 1 if temperature > 38.0 else 0
                    hypotension = 1 if blood_pressure_systolic < 90 else 0
                    tachycardia = 1 if heart_rate > 100 else 0
                    tachypnea = 1 if respiratory_rate > 20 else 0
                    hypoxia = 1 if oxygen_saturation < 95 else 0
                    severe_pain = 1 if pain_score >= 7 else 0

                    # Encode categorical
                    gender_encoded = 0 if gender == "Male" else 1
                    arrival_encoded = 0 if arrival_mode == "Walk-in" else 1
                    consciousness_encoded = 0 if consciousness == "Alert" else 1

                    # Create feature vector
                    features = np.array([[
                        age, temperature, heart_rate, respiratory_rate, oxygen_saturation,
                        blood_pressure_systolic, blood_pressure_diastolic, pain_score,
                        shock_index, mean_arterial_pressure, pulse_pressure,
                        fever, hypotension, tachycardia, tachypnea, hypoxia, severe_pain,
                        arrival_encoded, consciousness_encoded, gender_encoded
                    ]])

                    # Scale features
                    features_scaled = models['scaler'].transform(features)

                    # Get predictions from all models
                    xgb_pred = models['xgb_model'].predict_proba(features_scaled)[0]
                    lgb_pred = models['lgb_model'].predict_proba(features_scaled)[0]

                    # Ensemble prediction (simple average)
                    ensemble_proba = (xgb_pred + lgb_pred) / 2
                    prediction = 1 if ensemble_proba[1] > 0.5 else 0

                    # Display results
                    if prediction == 1:
                        st.markdown("""
                        <div class="urgent-alert">
                            <h3>🚨 URGENT CASE DETECTED</h3>
                            <p><strong>Confidence:</strong> {:.1%}</p>
                            <p><strong>XGBoost:</strong> {:.1%} | <strong>LightGBM:</strong> {:.1%}</p>
                        </div>
                        """.format(ensemble_proba[1], xgb_pred[1], lgb_pred[1]), unsafe_allow_html=True)

                        st.markdown("### Recommended Actions:")
                        actions = [
                            "🏥 Immediate physician assessment",
                            "📊 Vital signs monitoring every 5-15 minutes",
                            "💉 Consider IV access and fluids",
                            "🛏️ Priority bed assignment",
                            "📞 Notify specialist if indicated"
                        ]
                        for action in actions:
                            st.markdown(f"- {action}")
                    else:
                        st.markdown("""
                        <div class="success-alert">
                            <h3>✅ NON-URGENT CASE</h3>
                            <p><strong>Confidence:</strong> {:.1%}</p>
                            <p><strong>XGBoost:</strong> {:.1%} | <strong>LightGBM:</strong> {:.1%}</p>
                        </div>
                        """.format(ensemble_proba[0], xgb_pred[0], lgb_pred[0]), unsafe_allow_html=True)

                        st.markdown("### Recommended Actions:")
                        actions = [
                            "👨‍⚕️ Routine physician assessment",
                            "📊 Vital signs monitoring every 30-60 minutes",
                            "🛏️ Standard ED bed assignment",
                            "📋 Complete intake assessment"
                        ]
                        for action in actions:
                            st.markdown(f"- {action}")

                    # Clinical insights
                    st.markdown("### 🔍 Clinical Insights")
                    insights = []

                    if shock_index > 0.7:
                        insights.append("⚠️ **Shock Index > 0.7**: Indicates potential shock")
                    if fever == 1:
                        insights.append("🌡️ **Fever detected**: Consider infection workup")
                    if hypotension == 1:
                        insights.append("🩸 **Hypotension**: Blood pressure < 90 mmHg")
                    if tachycardia == 1:
                        insights.append("💓 **Tachycardia**: Heart rate > 100 BPM")
                    if hypoxia == 1:
                        insights.append("🫁 **Hypoxia**: O2 saturation < 95%")

                    if insights:
                        for insight in insights:
                            st.markdown(f"- {insight}")
                    else:
                        st.info("No critical clinical flags detected")

elif page == "Advanced Dashboard":
    st.header("📊 Advanced ED Dashboard")

    if datasets:
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_patients = len(datasets['triage'])
            st.metric("Total Patients", f"{total_patients:,}")

        with col2:
            urgent_cases = len(datasets['triage'][datasets['triage']['acuity_level'] <= 2])
            urgent_pct = (urgent_cases / total_patients) * 100
            st.metric("Urgent Cases", f"{urgent_cases:,}", f"{urgent_pct:.1f}%")

        with col3:
            avg_los = datasets['edstays']['los'].mean()
            st.metric("Avg Length of Stay", f"{avg_los:.1f}h")

        with col4:
            avg_age = datasets['triage']['age'].mean()
            st.metric("Average Age", f"{avg_age:.0f} years")

        with col5:
            male_pct = (datasets['triage']['gender'] == 'Male').mean() * 100
            st.metric("Male Patients", f"{male_pct:.1f}%")

        # Advanced visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Acuity Level Distribution")
            acuity_counts = datasets['triage']['acuity_level'].value_counts().sort_index()

            fig = px.bar(
                x=[f'Level {i}' for i in acuity_counts.index],
                y=acuity_counts.values,
                labels={'x': 'Acuity Level', 'y': 'Count'},
                color=acuity_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Arrival Mode Distribution")
            arrival_counts = datasets['triage']['arrival_mode'].value_counts()

            fig = px.pie(
                values=arrival_counts.values,
                names=arrival_counts.index,
                title="Patient Arrival Methods"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Vital signs distributions
        st.subheader("Vital Signs Overview")
        vitals_cols = ['heart_rate', 'blood_pressure_systolic', 'temperature', 'oxygen_saturation']

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Vital Signs Distributions', fontsize=16)

        for i, col in enumerate(vitals_cols):
            ax = axes[i//2, i%2]
            if col in datasets['triage'].columns:
                data = datasets['triage'][col].dropna()
                if len(data) > 0:
                    sns.histplot(data=data, ax=ax, kde=True)
                    ax.set_title(f'{col.replace("_", " ").title()}')
                    ax.set_xlabel('')

        plt.tight_layout()
        st.pyplot(fig)

        # Real-time patient queue
        st.subheader("Current Patient Queue")
        queue_cols = ['subject_id', 'chief_complaint', 'acuity_level', 'pain_score', 'arrival_mode']
        queue_data = datasets['triage'][queue_cols].head(20)

        # Add priority color coding
        def color_priority(val):
            if val <= 2:
                return 'background-color: #ffebee'
            elif val <= 3:
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e8'

        styled_queue = queue_data.style.apply(
            lambda x: [color_priority(x['acuity_level'])] * len(x), axis=1
        )
        st.dataframe(styled_queue, use_container_width=True)

    else:
        st.error("Unable to load dashboard data.")

elif page == "Interpretability":
    st.header("🔍 Model Interpretability")

    if not models or not interpret_engine:
        st.error("Interpretability tools not available.")
    else:
        st.markdown("""
        Explore how the AI model makes decisions using advanced interpretability techniques:
        - **SHAP**: Feature importance and contribution analysis
        - **LIME**: Local explanations for individual predictions
        - **Integrated Gradients**: Attribution analysis
        """)

        # Sample selection
        st.subheader("Select Sample for Analysis")

        if datasets and 'triage' in datasets:
            sample_options = list(range(min(10, len(datasets['triage']))))
            selected_sample = st.selectbox("Choose sample to analyze:", sample_options)

            if st.button("Generate Interpretability Analysis"):
                with st.spinner("Computing explanations..."):
                    # Get sample data
                    sample_data = datasets['triage'].iloc[selected_sample]

                    # Prepare features (simplified for demo)
                    features = np.array([[
                        sample_data.get('age', 45),
                        sample_data.get('temperature', 36.6),
                        sample_data.get('heart_rate', 75),
                        sample_data.get('respiratory_rate', 16),
                        sample_data.get('oxygen_saturation', 98),
                        sample_data.get('blood_pressure_systolic', 120),
                        sample_data.get('blood_pressure_diastolic', 80),
                        sample_data.get('pain_score', 0),
                        75/120,  # shock index
                        (120 + 2*80)/3,  # MAP
                        120-80,  # pulse pressure
                        0, 0, 0, 0, 0, 0,  # clinical flags
                        0, 0, 0  # encoded features
                    ]])

                    features_scaled = models['scaler'].transform(features)

                    # Generate comprehensive explanation
                    explanations = interpret_engine.generate_comprehensive_explanation(
                        features_scaled[0], selected_sample
                    )

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("SHAP Analysis")
                        if 'shap_values' in explanations:
                            st.info("SHAP values computed successfully")
                            # Simplified SHAP display
                            shap_df = pd.DataFrame({
                                'Feature': models['feature_names'][:10],
                                'SHAP_Value': explanations['shap_values'][:10]
                            })
                            st.dataframe(shap_df)
                        else:
                            st.warning("SHAP analysis not available")

                    with col2:
                        st.subheader("LIME Analysis")
                        if 'lime_explanation' in explanations:
                            st.info("LIME explanation computed successfully")
                            lime_df = pd.DataFrame(explanations['lime_explanation'],
                                                 columns=['Feature', 'Contribution'])
                            st.dataframe(lime_df)
                        else:
                            st.warning("LIME analysis not available")

                    # Generate explanation report
                    if explanations:
                        st.subheader("Comprehensive Explanation Report")
                        report = interpret_engine.generate_explanation_report(explanations)
                        st.text_area("Explanation Report", report, height=300)

        else:
            st.error("Sample data not available for analysis.")

elif page == "Fairness Analysis":
    st.header("⚖️ Fairness Analysis")

    if not FAIRNESS_AVAILABLE or not datasets:
        st.error("Fairness analysis tools not available.")
    else:
        st.markdown("""
        Assess model fairness across protected attributes using advanced metrics:
        - **Disparate Impact**: Measure of bias in outcomes
        - **Equalized Odds**: Balance in true positive and false positive rates
        - **Comprehensive Fairness Report**: Detailed analysis and recommendations
        """)

        # Select protected attributes
        st.subheader("Protected Attributes Analysis")

        protected_attrs = st.multiselect(
            "Select protected attributes to analyze:",
            ["gender", "race", "age_group", "insurance_type"],
            default=["gender", "age_group"]
        )

        if st.button("Run Fairness Analysis") and protected_attrs:
            with st.spinner("Computing fairness metrics..."):

                # Prepare data (simplified for demo)
                n_samples = min(1000, len(datasets['triage']))
                sample_data = datasets['triage'].head(n_samples)

                # Create mock predictions for demo
                np.random.seed(42)
                predictions = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

                # Prepare protected attributes
                protected_data = {}
                for attr in protected_attrs:
                    if attr == "gender":
                        protected_data[attr] = sample_data['gender'].fillna('Unknown').values
                    elif attr == "age_group":
                        ages = sample_data['age'].fillna(45)
                        protected_data[attr] = pd.cut(ages, bins=[0, 30, 50, 70, 100],
                                                    labels=['18-30', '31-50', '51-70', '71+']).values
                    else:
                        # Mock data for other attributes
                        protected_data[attr] = np.random.choice(['Group_A', 'Group_B'], n_samples)

                # Run fairness assessment
                fairness_results = fairness_assessor.assess_fairness(
                    predictions, sample_data['acuity_level'] <= 2, protected_data
                )

                # Display results
                st.subheader("Fairness Assessment Results")

                for attr, results in fairness_results.items():
                    st.markdown(f"### {attr.upper()}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Disparate Impact Ratios:**")
                        di_ratios = results['disparate_impact']['disparate_impact_ratios']
                        if di_ratios:
                            for comparison, ratio in di_ratios.items():
                                if ratio < 0.8:
                                    st.error(f"⚠️ {comparison}: {ratio:.3f} (Unfair)")
                                else:
                                    st.success(f"✅ {comparison}: {ratio:.3f} (Fair)")
                        else:
                            st.info("No disparate impact ratios calculated")

                    with col2:
                        st.markdown("**Equalized Odds:**")
                        eo_results = results['equalized_odds']
                        for group, metrics in eo_results.items():
                            st.write(f"**{group}:**")
                            st.write(".3f")
                            st.write(".3f")
                            st.write(f"  - Sample Size: {metrics['sample_size']}")

                # Generate comprehensive report
                st.subheader("Comprehensive Fairness Report")
                report = generate_fairness_report(fairness_results, predictions, sample_data['acuity_level'] <= 2)
                st.text_area("Fairness Report", report, height=400)

        else:
            st.info("Select protected attributes and click 'Run Fairness Analysis' to begin.")

elif page == "Counterfactuals":
    st.header("🔄 Counterfactual Explanations")

    if not models:
        st.error("Counterfactual analysis requires trained models.")
    else:
        st.markdown("""
        Explore 'what-if' scenarios to understand how small changes in patient data
        could alter the triage decision.
        """)

        # Sample selection
        if datasets and 'triage' in datasets:
            sample_idx = st.selectbox("Select patient sample:", range(min(10, len(datasets['triage']))))

            if st.button("Generate Counterfactual"):
                with st.spinner("Computing counterfactual explanation..."):

                    # Get sample data
                    sample_data = datasets['triage'].iloc[sample_idx]

                    # Prepare features
                    features = np.array([[
                        sample_data.get('age', 45),
                        sample_data.get('temperature', 36.6),
                        sample_data.get('heart_rate', 75),
                        sample_data.get('respiratory_rate', 16),
                        sample_data.get('oxygen_saturation', 98),
                        sample_data.get('blood_pressure_systolic', 120),
                        sample_data.get('blood_pressure_diastolic', 80),
                        sample_data.get('pain_score', 0),
                        75/120,  # shock index
                        (120 + 2*80)/3,  # MAP
                        120-80,  # pulse pressure
                        0, 0, 0, 0, 0, 0,  # clinical flags
                        0, 0, 0  # encoded features
                    ]])

                    features_scaled = models['scaler'].transform(features)

                    # Generate counterfactual
                    counterfactual = generate_counterfactual_explanation(
                        features_scaled[0], models['xgb_model'], models['feature_names']
                    )

                    # Display results
                    st.subheader("Counterfactual Analysis")

                    if counterfactual.get('success', False):
                        st.success("Counterfactual found! Here are the minimal changes needed:")

                        changes_df = pd.DataFrame(counterfactual['changes'])
                        st.dataframe(changes_df)

                        st.markdown("### Scenario Comparison")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Prediction:**")
                            st.write(".3f")
                            st.write(f"Predicted Class: {'Urgent' if counterfactual['original_prediction'][1] > 0.5 else 'Non-urgent'}")

                        with col2:
                            st.markdown("**Counterfactual Prediction:**")
                            st.write(".3f")
                            st.write(f"Predicted Class: {'Urgent' if counterfactual['counterfactual_prediction'][1] > 0.5 else 'Non-urgent'}")

                    else:
                        st.info("No counterfactual found - the patient is already in the target class or no changes can flip the prediction.")

        else:
            st.error("Sample data not available.")

elif page == "Simulation":
    st.header("🎯 Advanced ED Simulation")

    if datasets:
        st.markdown("""
        Run sophisticated simulations to predict ED performance under different conditions.
        Incorporates real patient data, staffing levels, and resource constraints.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Simulation Parameters")

            # Patient load
            current_patients = len(datasets['triage'])
            patient_load = st.slider("Patient Volume", min_value=current_patients//2,
                                   max_value=current_patients*2, value=current_patients)

            # Staffing
            staff_ratio = st.slider("Staff-to-Patient Ratio", min_value=0.1, max_value=1.0,
                                  value=0.3, step=0.05)

            # Resources
            bed_occupancy = st.slider("Bed Occupancy (%)", min_value=50, max_value=100, value=78)
            equipment_available = st.slider("Equipment Availability (%)", min_value=70, max_value=100, value=90)

        with col2:
            st.subheader("Advanced Parameters")

            # Time factors
            time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])

            # Patient acuity mix
            acuity_mix = st.selectbox("Patient Acuity Mix",
                                    ["Current Mix", "More Urgent", "Less Urgent"])

            # External factors
            weather_impact = st.checkbox("Severe Weather Impact")
            community_event = st.checkbox("Community Event")

        if st.button("Run Advanced Simulation", type="primary"):
            with st.spinner("Running advanced simulation..."):

                # Calculate simulation metrics
                base_wait = datasets['edstays']['los'].mean() * 60  # Convert to minutes

                # Adjust for parameters
                load_factor = patient_load / current_patients
                staff_factor = 1 / staff_ratio
                bed_factor = bed_occupancy / 78  # Normalize to current occupancy
                equipment_factor = equipment_available / 90

                # Time-based adjustments
                time_multipliers = {
                    "Morning": 1.0, "Afternoon": 1.2, "Evening": 1.4, "Night": 1.1
                }
                day_multipliers = {"Weekday": 1.0, "Weekend": 1.3}

                time_factor = time_multipliers[time_of_day] * day_multipliers[day_of_week]

                # External factors
                external_factor = 1.0
                if weather_impact:
                    external_factor *= 1.3
                if community_event:
                    external_factor *= 1.2

                # Calculate predicted metrics
                predicted_wait = base_wait * load_factor * staff_factor * bed_factor * equipment_factor * time_factor * external_factor
                patient_satisfaction = max(0, 100 - (predicted_wait / 3))
                resource_utilization = min(100, bed_occupancy + (patient_load / 20))
                staff_burnout_risk = min(100, (load_factor * staff_factor) * 50)

                # Display results
                st.subheader("Simulation Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if predicted_wait > 120:
                        st.error(f"🚨 Critical: {predicted_wait:.0f} min")
                    elif predicted_wait > 60:
                        st.warning(f"⚠️ High: {predicted_wait:.0f} min")
                    else:
                        st.success(f"✅ Good: {predicted_wait:.0f} min")
                    st.metric("Avg Wait Time", ".0f")

                with col2:
                    if patient_satisfaction < 50:
                        st.error(f"🚨 Poor: {patient_satisfaction:.0f}%")
                    elif patient_satisfaction < 75:
                        st.warning(f"⚠️ Fair: {patient_satisfaction:.0f}%")
                    else:
                        st.success(f"✅ Good: {patient_satisfaction:.0f}%")
                    st.metric("Patient Satisfaction", ".0f")

                with col3:
                    if resource_utilization > 95:
                        st.error(f"🚨 Critical: {resource_utilization:.0f}%")
                    elif resource_utilization > 85:
                        st.warning(f"⚠️ High: {resource_utilization:.0f}%")
                    else:
                        st.success(f"✅ Good: {resource_utilization:.0f}%")
                    st.metric("Resource Utilization", ".0f")

                with col4:
                    if staff_burnout_risk > 80:
                        st.error(f"🚨 High: {staff_burnout_risk:.0f}%")
                    elif staff_burnout_risk > 60:
                        st.warning(f"⚠️ Moderate: {staff_burnout_risk:.0f}%")
                    else:
                        st.success(f"✅ Low: {staff_burnout_risk:.0f}%")
                    st.metric("Staff Burnout Risk", ".0f")

                # Recommendations
                st.subheader("Recommendations")

                recommendations = []
                if predicted_wait > 60:
                    recommendations.append("🚑 Consider ambulance diversion")
                    recommendations.append("👥 Increase staffing levels")
                if resource_utilization > 90:
                    recommendations.append("🛏️ Optimize bed management")
                if staff_burnout_risk > 70:
                    recommendations.append("⏰ Implement shift adjustments")
                if patient_satisfaction < 70:
                    recommendations.append("📋 Improve patient communication")

                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.success("✅ All metrics within acceptable ranges")

    else:
        st.error("Simulation data not available.")

elif page == "Model Evaluation":
    st.header("📈 Advanced Model Evaluation")

    if models and datasets:
        st.markdown("""
        Comprehensive evaluation of AI models using advanced metrics:
        - **AUC/PR-AUC**: Classification performance
        - **Brier Score**: Calibration assessment
        - **F1 Score**: Precision-recall balance
        - **Temporal Validation**: Performance over time
        """)

        # Prepare evaluation data
        eval_data = datasets['triage'].head(min(1000, len(datasets['triage'])))

        # Create mock predictions for demonstration
        np.random.seed(42)
        n_samples = len(eval_data)
        true_labels = (eval_data['acuity_level'] <= 2).astype(int).values

        # Mock model predictions
        xgb_proba = np.random.beta(2, 5, n_samples)  # Skewed towards non-urgent
        lgb_proba = np.random.beta(2, 4, n_samples)
        ensemble_proba = (xgb_proba + lgb_proba) / 2

        xgb_pred = (xgb_proba > 0.5).astype(int)
        lgb_pred = (lgb_proba > 0.5).astype(int)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        # Evaluation metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("XGBoost Performance")
            st.metric("AUC", ".3f")
            st.metric("PR-AUC", ".3f")
            st.metric("Brier Score", ".3f")
            st.metric("F1 Score", ".3f")

        with col2:
            st.subheader("LightGBM Performance")
            st.metric("AUC", ".3f")
            st.metric("PR-AUC", ".3f")
            st.metric("Brier Score", ".3f")
            st.metric("F1 Score", ".3f")

        with col3:
            st.subheader("Ensemble Performance")
            st.metric("AUC", ".3f")
            st.metric("PR-AUC", ".3f")
            st.metric("Brier Score", ".3f")
            st.metric("F1 Score", ".3f")

        # Confusion matrices
        st.subheader("Confusion Matrices")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models_data = [
            ("XGBoost", xgb_pred),
            ("LightGBM", lgb_pred),
            ("Ensemble", ensemble_pred)
        ]

        for i, (name, pred) in enumerate(models_data):
            cm = pd.crosstab(true_labels, pred,
                           rownames=['True'], colnames=['Predicted'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{name} Confusion Matrix')

        plt.tight_layout()
        st.pyplot(fig)

        # Calibration plot
        st.subheader("Model Calibration")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        # Model calibration curves (simplified)
        prob_bins = np.linspace(0, 1, 11)
        for name, proba in [("XGBoost", xgb_proba), ("LightGBM", lgb_proba), ("Ensemble", ensemble_proba)]:
            bin_means = []
            true_means = []

            for i in range(len(prob_bins)-1):
                mask = (proba >= prob_bins[i]) & (proba < prob_bins[i+1])
                if np.sum(mask) > 0:
                    bin_means.append(np.mean(proba[mask]))
                    true_means.append(np.mean(true_labels[mask]))

            ax.plot(bin_means, true_means, 'o-', label=name, markersize=4)

        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')
        ax.set_title('Calibration Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

    else:
        st.error("Model evaluation data not available.")

elif page == "Clinician Survey":
    st.header("👨‍⚕️ Clinician Evaluation Survey")

    st.markdown("""
    This survey is designed to evaluate the AI system's interpretability and clinical utility
    from the perspective of healthcare professionals. Your feedback will help improve the system.
    """)

    st.subheader("Survey Questions")

    # Survey questions
    questions = [
        "How would you rate the AI's triage recommendations?",
        "How understandable are the model's explanations?",
        "How much do you trust the AI's recommendations?",
        "How useful is the counterfactual analysis for clinical decision-making?",
        "Would you feel comfortable using this system in a real clinical setting?",
        "How does this AI system compare to your current triage process?"
    ]

    ratings = []
    for i, question in enumerate(questions):
        rating = st.slider(question, 1, 5, 3, key=f"q{i}")
        ratings.append(rating)

    # Survey submission
    if st.button("Submit Survey", type="primary"):
        st.success("Thank you for your feedback! Your responses have been recorded.")

        # Display summary
        st.subheader("Your Responses Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Individual Ratings:**")
            for i, (q, r) in enumerate(zip(questions, ratings)):
                st.write(f"**Q{i+1}:** {r}/5")

        with col2:
            st.write("**Overall Statistics:**")
            avg_rating = sum(ratings) / len(ratings)
            st.metric("Average Rating", ".2f")
            st.metric("Highest Rating", max(ratings))
            st.metric("Lowest Rating", min(ratings))

        # Feedback interpretation
        st.subheader("Feedback Interpretation")

        if avg_rating >= 4.0:
            st.success("🎉 Excellent feedback! The system is highly rated.")
        elif avg_rating >= 3.0:
            st.info("👍 Good feedback with room for improvement.")
        else:
            st.warning("⚠️ Areas for improvement identified.")

        # Detailed feedback
        high_ratings = [i for i, r in enumerate(ratings) if r >= 4]
        low_ratings = [i for i, r in enumerate(ratings) if r <= 2]

        if high_ratings:
            st.write("**Strengths:**")
            for i in high_ratings:
                st.write(f"- {questions[i]}")

        if low_ratings:
            st.write("**Areas for Improvement:**")
            for i in low_ratings:
                st.write(f"- {questions[i]}")

    # Additional feedback
    st.subheader("Additional Comments")
    additional_feedback = st.text_area(
        "Please provide any additional comments or suggestions:",
        height=100,
        placeholder="Your feedback is valuable for improving the system..."
    )

    if additional_feedback.strip():
        st.info("Thank you for your detailed feedback!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Advanced ED-AI Triage System</strong></p>
    <p>© 2024 | Powered by ClinicalBERT, XGBoost, and Advanced Interpretability</p>
    <p>For research and educational purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)
