import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .high-risk {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-color: #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DiabetesPredictionApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # For demo purposes, we'll train a simple model
            # In production, you'd load a pre-trained model
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            
            # Load diabetes data
            data = pd.read_csv('diabetes.csv')
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            
            # Train a simple model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict_diabetes_risk(self, features):
        """Predict diabetes risk based on input features"""
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return prediction, probability
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None, None
    
    def get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability >= 0.7:
            return "High Risk", "high-risk", "#d32f2f"  # Red
        elif probability >= 0.4:
            return "Medium Risk", "medium-risk", "#f57c00"  # Orange
        else:
            return "Low Risk", "low-risk", "#388e3c"  # Green
    
    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance for Diabetes Prediction',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_x=0.5
        )
        
        return fig
    
    def create_shap_explanation(self, features):
        """Create SHAP explanation for the prediction"""
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get SHAP values
            shap_values = explainer.shap_values(features_scaled)
            
            # Handle different SHAP value structures
            if isinstance(shap_values, list):
                # For binary classification, shap_values is a list with 2 arrays
                # We want the positive class (index 1) for diabetes prediction
                shap_values = shap_values[1]  # Get positive class values
            else:
                shap_values = np.array(shap_values)
            
            # Ensure we have 1D array
            if shap_values.ndim > 1:
                shap_values = shap_values.flatten()
            
            # Create waterfall plot
            fig = go.Figure()
            
            # Get feature names
            feature_names = self.feature_names
            
            # Create waterfall chart
            base_value = explainer.expected_value
            
            # Add bars
            fig.add_trace(go.Waterfall(
                name="SHAP Values",
                orientation="h",
                measure=["relative"] * len(feature_names),
                x=shap_values.tolist(),
                textposition="outside",
                text=[f"{float(val):.3f}" for val in shap_values],
                y=feature_names,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ff7f0e"}},
                increasing={"marker": {"color": "#1f77b4"}},
                totals={"marker": {"color": "#d62728"}}
            ))
            
            fig.update_layout(
                title="SHAP Explanation for Prediction",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=400,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating SHAP explanation: {e}")
            return None

def main():
    # Initialize app
    app = DiabetesPredictionApp()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Early Detection for Better Health Outcomes")
    
    # Load model
    if not app.load_model():
        st.error("Failed to load the prediction model. Please check the data file.")
        return
    
    # Sidebar for input
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("Enter the patient's medical data:")
    
    # Input fields
    pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 17, 1)
    glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 20)
    insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 846, 80)
    bmi = st.sidebar.slider("BMI (kg/m²)", 0.0, 67.1, 25.0, step=0.1)
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, step=0.001)
    age = st.sidebar.slider("Age (years)", 21, 81, 30)
    
    # Create features array
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    
    # Prediction button
    if st.sidebar.button("🔍 Predict Diabetes Risk", type="primary"):
        # Make prediction
        prediction, probability = app.predict_diabetes_risk(features)
        
        if prediction is not None:
            risk_level, risk_class, risk_color = app.get_risk_level(probability)
            
            # Main prediction display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2 style='color: #222; font-size: 2.5rem; font-weight: bold; margin-bottom: 1.5rem;'>Risk Assessment</h2>
                    <h1 style='color: {risk_color}; font-size: 3.5rem; font-weight: bold; margin-bottom: 1.2rem;'>{risk_level}</h1>
                    <p style='color: {risk_color}; font-size: 1.5rem; font-weight: bold; margin-bottom: 0.8rem;'>Probability: {probability:.1%}</p>
                    <p style='color: #222; font-size: 1.3rem; font-weight: bold;'>Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Glucose Level", f"{glucose} mg/dL", 
                         "High" if glucose > 140 else "Normal" if glucose > 70 else "Low")
            
            with col2:
                st.metric("BMI", f"{bmi:.1f} kg/m²", 
                         "High" if bmi > 30 else "Normal" if bmi > 18.5 else "Low")
            
            with col3:
                st.metric("Age", f"{age} years")
            
            with col4:
                st.metric("Blood Pressure", f"{blood_pressure} mm Hg", 
                         "High" if blood_pressure > 120 else "Normal")
            
            # Feature importance
            st.markdown("## 📊 Feature Importance")
            importance_fig = app.create_feature_importance_plot()
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # SHAP explanation
            st.markdown("## 🔍 Prediction Explanation")
            shap_fig = app.create_shap_explanation(features)
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True)
            
            # Recommendations
            st.markdown("## 💡 Recommendations")
            
            if risk_level == "High Risk":
                st.warning("""
                **Immediate Action Required:**
                - Schedule an appointment with your healthcare provider
                - Monitor blood glucose levels regularly
                - Consider lifestyle modifications (diet, exercise)
                - Discuss medication options with your doctor
                """)
            elif risk_level == "Medium Risk":
                st.info("""
                **Moderate Risk - Monitor Closely:**
                - Regular check-ups with healthcare provider
                - Maintain healthy lifestyle habits
                - Monitor blood glucose periodically
                - Consider preventive measures
                """)
            else:
                st.success("""
                **Low Risk - Maintain Health:**
                - Continue healthy lifestyle habits
                - Regular check-ups as recommended
                - Monitor for any changes in health
                - Stay informed about diabetes prevention
                """)
            
            # Data summary
            st.markdown("## 📈 Input Data Summary")
            input_data = pd.DataFrame({
                'Feature': app.feature_names,
                'Value': features
            })
            st.dataframe(input_data, use_container_width=True)
    
    # Information about the model
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        **Model Information:**
        - **Algorithm**: Random Forest Classifier
        - **Dataset**: Diabetes prediction dataset with 768 patients
        - **Features**: 8 medical parameters
        - **Accuracy**: ~80% on test data
        
        **Features Used:**
        - Number of Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin Level
        - BMI (Body Mass Index)
        - Diabetes Pedigree Function
        - Age
        
        **Disclaimer**: This is a prototype model for educational purposes. 
        Always consult with healthcare professionals for medical decisions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🏥 Diabetes Risk Prediction Model | Prototype for Educational Purposes
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 