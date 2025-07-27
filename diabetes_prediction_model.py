import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DiabetesPredictionModel:
    def __init__(self, data_path='diabetes.csv'):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("🔍 Loading Diabetes Dataset...")
        self.data = pd.read_csv(self.data_path)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print("\n📊 Dataset Info:")
        print(self.data.info())
        
        print("\n📈 Basic Statistics:")
        print(self.data.describe())
        
        print("\n🎯 Target Distribution:")
        print(self.data['Outcome'].value_counts())
        print(f"Diabetes Rate: {self.data['Outcome'].mean():.2%}")
        
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n🔬 Exploratory Data Analysis")
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Check for zero values (which might indicate missing data)
        print("\nZero Values (potential missing data):")
        zero_counts = (self.data == 0).sum()
        print(zero_counts)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Distribution plots for each feature
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        features = self.data.columns[:-1]  # Exclude Outcome
        
        for i, feature in enumerate(features):
            row = i // 4
            col = i % 4
            axes[row, col].hist(self.data[feature], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{feature} Distribution', fontweight='bold')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Box plots for features by outcome
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        for i, feature in enumerate(features):
            row = i // 4
            col = i % 4
            self.data.boxplot(column=feature, by='Outcome', ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by Outcome', fontweight='bold')
            axes[row, col].set_xlabel('Outcome')
            axes[row, col].set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig('feature_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n🛠️ Data Preprocessing")
        
        # Handle missing values (zeros in some features might indicate missing data)
        # For medical data, we'll replace zeros with median for certain features
        features_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for feature in features_to_fix:
            # Replace zeros with median (only for non-zero median)
            median_val = self.data[feature].median()
            if median_val > 0:
                self.data[feature] = self.data[feature].replace(0, median_val)
        
        # Separate features and target
        self.X = self.data.drop('Outcome', axis=1)
        self.y = self.data['Outcome']
        self.feature_names = self.X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Features: {self.feature_names}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def feature_engineering(self):
        """Perform feature engineering"""
        print("\n🔧 Feature Engineering")
        
        # Create interaction features
        self.X_train['Glucose_BMI'] = self.X_train['Glucose'] * self.X_train['BMI']
        self.X_train['Age_BMI'] = self.X_train['Age'] * self.X_train['BMI']
        self.X_train['Glucose_Age'] = self.X_train['Glucose'] * self.X_train['Age']
        
        self.X_test['Glucose_BMI'] = self.X_test['Glucose'] * self.X_test['BMI']
        self.X_test['Age_BMI'] = self.X_test['Age'] * self.X_test['BMI']
        self.X_test['Glucose_Age'] = self.X_test['Glucose'] * self.X_test['Age']
        
        # Update feature names
        self.feature_names = self.X_train.columns.tolist()
        
        # Scale the new features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"New features added: Glucose_BMI, Age_BMI, Glucose_Age")
        print(f"Total features: {len(self.feature_names)}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n🤖 Training Machine Learning Models")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n📊 Model Evaluation")
        
        # Create comparison table
        comparison_data = []
        for name, metrics in self.models.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC': f"{metrics['auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        for name, metrics in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, metrics['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, metrics) in enumerate(self.models.items()):
            cm = confusion_matrix(self.y_test, metrics['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} Confusion Matrix', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def feature_importance_analysis(self):
        """Analyze feature importance for the best model"""
        print("\n🎯 Feature Importance Analysis")
        
        # Use Random Forest for feature importance
        rf_model = self.models['Random Forest']['model']
        
        # Get feature importance
        importance = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        print(feature_importance_df)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(10), 
                   x='Importance', y='Feature', palette='viridis')
        plt.title('Top 10 Most Important Features', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        print("\n🔧 Hyperparameter Tuning")
        
        # Tune Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        rf_grid.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best Random Forest parameters: {rf_grid.best_params_}")
        print(f"Best Random Forest F1 score: {rf_grid.best_score_:.4f}")
        
        # Update the best model
        self.models['Random Forest (Tuned)'] = {
            'model': rf_grid.best_estimator_,
            'accuracy': accuracy_score(self.y_test, rf_grid.predict(self.X_test_scaled)),
            'precision': precision_score(self.y_test, rf_grid.predict(self.X_test_scaled)),
            'recall': recall_score(self.y_test, rf_grid.predict(self.X_test_scaled)),
            'f1': f1_score(self.y_test, rf_grid.predict(self.X_test_scaled)),
            'auc': roc_auc_score(self.y_test, rf_grid.predict_proba(self.X_test_scaled)[:, 1]),
            'predictions': rf_grid.predict(self.X_test_scaled),
            'probabilities': rf_grid.predict_proba(self.X_test_scaled)[:, 1]
        }
        
        return rf_grid.best_estimator_
    
    def generate_insights(self):
        """Generate insights and recommendations"""
        print("\n💡 Key Insights and Recommendations")
        
        # Get the best model
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['f1'])
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   - F1 Score: {best_model['f1']:.4f}")
        print(f"   - Accuracy: {best_model['accuracy']:.4f}")
        print(f"   - Precision: {best_model['precision']:.4f}")
        print(f"   - Recall: {best_model['recall']:.4f}")
        print(f"   - AUC: {best_model['auc']:.4f}")
        
        # Feature importance insights
        feature_importance_df = self.feature_importance_analysis()
        top_features = feature_importance_df.head(5)['Feature'].tolist()
        
        print(f"\n🎯 Top 5 Most Important Features:")
        for i, feature in enumerate(top_features, 1):
            print(f"   {i}. {feature}")
        
        # Business insights
        print(f"\n📈 Business Insights:")
        print(f"   - The model can predict diabetes with {best_model['accuracy']:.1%} accuracy")
        print(f"   - Key risk factors: {', '.join(top_features[:3])}")
        print(f"   - Model precision: {best_model['precision']:.1%} (low false positives)")
        print(f"   - Model recall: {best_model['recall']:.1%} (identifies most cases)")
        
        return best_model_name, best_model

def main():
    """Main function to run the complete analysis"""
    print("🧠 Diabetes Prediction Model - Complete Analysis")
    print("=" * 60)
    
    # Initialize the model
    model = DiabetesPredictionModel()
    
    # Load and explore data
    model.load_data()
    model.explore_data()
    
    # Preprocess data
    model.preprocess_data()
    
    # Feature engineering
    model.feature_engineering()
    
    # Train models
    model.train_models()
    
    # Evaluate models
    comparison_df = model.evaluate_models()
    
    # Feature importance
    model.feature_importance_analysis()
    
    # Hyperparameter tuning
    model.hyperparameter_tuning()
    
    # Generate insights
    best_model_name, best_model = model.generate_insights()
    
    print("\n✅ Analysis Complete!")
    print("📁 Generated files:")
    print("   - correlation_matrix.png")
    print("   - feature_distributions.png")
    print("   - feature_boxplots.png")
    print("   - roc_curves.png")
    print("   - confusion_matrices.png")
    print("   - feature_importance.png")
    
    return model, comparison_df

if __name__ == "__main__":
    model, results = main() 