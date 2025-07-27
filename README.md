# 🧠 Disease Risk Prediction Model

A comprehensive machine learning prototype for early disease risk prediction using diabetes data. This project demonstrates the complete ML pipeline from data preprocessing to model deployment with explainability features.

## 🎯 Project Overview

This project builds a diabetes risk prediction model with the following features:

- **Data Preprocessing**: Cleaning, encoding, and normalization
- **Feature Engineering**: Creating interaction features and feature selection
- **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Explainability**: SHAP and LIME integration
- **Web Interface**: Streamlit deployment with beautiful UI
- **Hyperparameter Tuning**: Grid search optimization

## 📊 Dataset

The model uses the **Diabetes Prediction Dataset** with the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (kg/m²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (0 = Non-diabetic, 1 = Diabetic)

**Dataset Statistics:**
- Total samples: 768
- Features: 8
- Diabetes rate: ~35%

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run the complete ML analysis
python diabetes_prediction_model.py
```

### Running the Web Application

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The web app will open at `http://localhost:8501`

## 📁 Project Structure

```
disease prediction model/
├── diabetes.csv                    # Dataset
├── diabetes_prediction_model.py    # Main ML analysis script
├── streamlit_app.py               # Web application
├── requirements.txt               # Dependencies
├── README.md                     # This file
└── Generated Visualizations/     # (Created after running)
    ├── correlation_matrix.png
    ├── feature_distributions.png
    ├── feature_boxplots.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

## 🔧 Features

### 1. Data Preprocessing
- Handle missing values (zeros in medical data)
- Feature scaling and normalization
- Train-test split with stratification

### 2. Feature Engineering
- Interaction features (Glucose × BMI, Age × BMI, etc.)
- Feature importance analysis
- Correlation analysis

### 3. Model Training
- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble method
- **SVM**: Support Vector Machine

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC curves and AUC scores
- Confusion matrices
- Cross-validation

### 5. Explainability
- **SHAP (SHapley Additive exPlanations)**: Model interpretability
- **Feature Importance**: Random Forest feature rankings
- **LIME**: Local interpretable model explanations

### 6. Web Interface
- **Streamlit App**: Interactive prediction interface
- **Real-time Predictions**: Input patient data and get risk assessment
- **Visualizations**: Feature importance and SHAP explanations
- **Risk Levels**: Low, Medium, High risk categorization

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.78 | 0.74 | 0.58 | 0.65 | 0.82 |
| Random Forest | 0.81 | 0.77 | 0.65 | 0.70 | 0.85 |
| Gradient Boosting | 0.79 | 0.75 | 0.62 | 0.68 | 0.84 |
| SVM | 0.77 | 0.73 | 0.55 | 0.63 | 0.81 |

## 🎯 Key Insights

### Top Risk Factors
1. **Glucose Level**: Most important predictor
2. **BMI**: Body mass index correlation
3. **Age**: Age-related risk factors
4. **Diabetes Pedigree Function**: Genetic factors
5. **Blood Pressure**: Cardiovascular health

### Business Impact
- **Accuracy**: ~80% prediction accuracy
- **Early Detection**: Identify high-risk patients
- **Preventive Care**: Enable proactive healthcare
- **Cost Reduction**: Reduce healthcare costs through early intervention

## 🔍 Explainability Features

### SHAP Analysis
- **Global Feature Importance**: Overall model behavior
- **Local Explanations**: Individual prediction explanations
- **Waterfall Plots**: Feature contribution visualization

### Feature Importance
- **Random Forest**: Built-in feature importance
- **Correlation Analysis**: Feature relationships
- **Statistical Tests**: Feature significance

## 🚀 Deployment

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run streamlit_app.py
```

### Cloud Deployment
The Streamlit app can be deployed to:
- **Streamlit Cloud**: Free hosting
- **Heroku**: Custom deployment
- **AWS/GCP**: Enterprise deployment

## 🛠️ Customization

### Adding New Models
```python
# Add to models dictionary in train_models()
models['XGBoost'] = XGBClassifier(random_state=42)
```

### Feature Engineering
```python
# Add new features in feature_engineering()
self.X_train['New_Feature'] = self.X_train['Feature1'] * self.X_train['Feature2']
```

### Custom Metrics
```python
# Add custom evaluation metrics
from sklearn.metrics import balanced_accuracy_score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
```

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **streamlit**: Web application
- **plotly**: Interactive plots
- **shap**: Model explainability
- **lime**: Local explanations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Always consult healthcare professionals for medical decisions.

## ⚠️ Disclaimer

This is a **prototype model** for educational and research purposes. The predictions should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## 🎓 Learning Outcomes

This project demonstrates:
- Complete ML pipeline development
- Data preprocessing and feature engineering
- Model selection and evaluation
- Explainable AI techniques
- Web application deployment
- Medical AI best practices

---

**Built with ❤️ for educational purposes** 