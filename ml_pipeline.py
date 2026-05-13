import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    def __init__(self, threshold=0.42):
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.lr_model = None
        self.rf_model = None
        self.best_model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.original_shape = None
        self.final_shape = None
        self.imputer = None
        
    def engineer_features(self, df):
        """Create engineered features"""
        df = df.copy()
        df['Balance_to_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
        df['Tenure_to_Age_Ratio'] = df['Tenure'] / (df['Age'] + 1)
        df['Age_Tenure_Interaction'] = df['Age'] * df['Tenure']
        df['Product_Interaction'] = df['NumOfProducts'] * df['IsActiveMember']
        return df
    
    def preprocess_data(self, df, target_col='Exited', fit=False):
        """Complete preprocessing pipeline"""
        self.original_shape = df.shape
        df = self.engineer_features(df)
        
        if target_col in df.columns:
            X = df.drop(target_col, axis=1)
            y = df[target_col]
        else:
            X = df
            y = None
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                else:
                    X[col] = 0
        
        if fit:
            self.imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            self.feature_columns = X.columns.tolist()
        else:
            if self.imputer is not None:
                X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
            else:
                imputer = SimpleImputer(strategy='median')
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.final_shape = X_scaled.shape
        return X_scaled, y
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression"""
        param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']}
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.lr_model = grid_search.best_estimator_
        return self.lr_model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.rf_model = grid_search.best_estimator_
        return self.rf_model
    
    def tune_threshold(self, model, X_val, y_val):
        """Optimize decision threshold"""
        y_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.30, 0.55, 0.01)
        best_f1 = 0
        best_threshold = 0.42
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        return self.threshold
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        return {
            'model_name': model_name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def select_best_model(self, X_test, y_test):
        """Select best model based on F1 score"""
        lr_metrics = self.evaluate_model(self.lr_model, X_test, y_test, 'Logistic Regression')
        rf_metrics = self.evaluate_model(self.rf_model, X_test, y_test, 'Random Forest')
        
        if rf_metrics['f1_score'] >= lr_metrics['f1_score']:
            self.best_model = self.rf_model
            best_metrics = rf_metrics
        else:
            self.best_model = self.lr_model
            best_metrics = lr_metrics
        
        return self.best_model, best_metrics, lr_metrics, rf_metrics
    
    def get_feature_importance(self):
        """Get feature importance from best model"""
        if self.best_model is None:
            return pd.DataFrame({'feature': [], 'importance': []})
        
        if isinstance(self.best_model, RandomForestClassifier):
            importance = self.best_model.feature_importances_
        elif isinstance(self.best_model, LogisticRegression):
            importance = np.abs(self.best_model.coef_[0])
        else:
            importance = np.zeros(len(self.feature_columns)) if self.feature_columns else np.array([])
        
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        if self.feature_columns and len(importance) == len(self.feature_columns):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({'feature': [], 'importance': []})
        
        return feature_importance
    
    def predict(self, features_df):
        """Make predictions"""
        X_scaled, _ = self.preprocess_data(features_df, fit=False)
        probability = self.best_model.predict_proba(X_scaled)[:, 1]
        prediction = (probability >= self.threshold).astype(int)
        return prediction, probability
    
    def get_risk_level(self, probability):
        if probability >= 0.7:
            return "Very High Risk 🔴", "danger"
        elif probability >= 0.5:
            return "High Risk 🟠", "warning"
        elif probability >= self.threshold:
            return "Medium Risk 🟡", "info"
        elif probability >= 0.2:
            return "Low Risk 🟢", "success"
        else:
            return "Very Low Risk ⚪", "secondary"
    
    def get_retention_action(self, probability):
        if probability >= 0.7:
            return "🚨 URGENT: Immediate retention call + loyalty bonus offer"
        elif probability >= 0.5:
            return "📞 Schedule retention call within 48 hours"
        elif probability >= self.threshold:
            return "📧 Send personalized email with exclusive offers"
        elif probability >= 0.2:
            return "👥 Regular engagement and satisfaction surveys"
        else:
            return "✅ Standard monitoring and service"
    
    def get_shape_info(self):
        return {
            'original_rows': self.original_shape[0] if self.original_shape else 0,
            'original_cols': self.original_shape[1] if self.original_shape else 0,
            'final_rows': self.final_shape[0] if self.final_shape else 0,
            'final_cols': self.final_shape[1] if self.final_shape else 0
        }
    
    def save_models(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        if self.best_model:
            joblib.dump(self.best_model, os.path.join(path, 'best_model.pkl'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        if self.imputer:
            joblib.dump(self.imputer, os.path.join(path, 'imputer.pkl'))
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(path, 'feature_columns.pkl'))
        joblib.dump(self.threshold, os.path.join(path, 'threshold.pkl'))
    
    def load_models(self, path='models/'):
        self.best_model = joblib.load(os.path.join(path, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        self.imputer = joblib.load(os.path.join(path, 'imputer.pkl'))
        self.label_encoders = joblib.load(os.path.join(path, 'label_encoders.pkl'))
        self.feature_columns = joblib.load(os.path.join(path, 'feature_columns.pkl'))
        self.threshold = joblib.load(os.path.join(path, 'threshold.pkl'))