from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

class Customer(db.Model):
    __tablename__ = 'customers'
    
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50), unique=True, nullable=False)
    credit_score = db.Column(db.Integer)
    geography = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    tenure = db.Column(db.Integer)
    balance = db.Column(db.Float)
    estimated_salary = db.Column(db.Float)
    num_of_products = db.Column(db.Integer)
    has_cr_card = db.Column(db.Integer)
    is_active_member = db.Column(db.Integer)
    
    # Engineered features
    balance_to_salary_ratio = db.Column(db.Float)
    tenure_to_age_ratio = db.Column(db.Float)
    age_tenure_interaction = db.Column(db.Float)
    product_interaction = db.Column(db.Float)
    
    # Predictions
    churn_prediction = db.Column(db.Integer)
    churn_probability = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)
    recommended_action = db.Column(db.String(500))

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    customer_id = db.Column(db.String(50))
    churn_prediction = db.Column(db.Integer)
    churn_probability = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelMetrics(db.Model):
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    auc_roc = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)