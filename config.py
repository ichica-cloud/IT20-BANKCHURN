import os

class Config:
    SECRET_KEY = 'dev-secret-key-change-in-production'
    
    # SQLite Database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///bank_churn.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    MODEL_PATH = 'models/best_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    CHURN_THRESHOLD = 0.42