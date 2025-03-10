from flask import Flask, render_template, request, jsonify
import pandas as pd
import string
import re
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Function for text preprocessing with fixed regex patterns
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Fixed escape sequence with r prefix
    text = re.sub(r"\\W", " ", text)     # Fixed escape sequence with r prefix
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Fixed escape sequence with r prefix
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # Fixed escape sequence with r prefix
    return text

# Load models and vectorizer
model_dir = 'models'

# Check if models exist, otherwise inform the user
if not os.path.exists(model_dir):
    print("Model directory not found! Please train models first.")
    models_loaded = False
else:
    try:
        # Load the vectorizer
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Load all models
        with open(os.path.join(model_dir, 'model_lr.pkl'), 'rb') as f:
            LR = pickle.load(f)
            
        with open(os.path.join(model_dir, 'model_dt.pkl'), 'rb') as f:
            DT = pickle.load(f)
            
        with open(os.path.join(model_dir, 'model_gb.pkl'), 'rb') as f:
            GB = pickle.load(f)
            
        with open(os.path.join(model_dir, 'model_rf.pkl'), 'rb') as f:
            RF = pickle.load(f)
        
        print("All models loaded successfully!")
        models_loaded = True
    except Exception as e:
        print(f"Error loading models: {e}")
        models_loaded = False

@app.route('/')
def index():
    return render_template('index.html', models_ready=models_loaded)

@app.route('/analyze')
def analyze():
    return render_template('analyze-page.html', models_ready=models_loaded)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def working():
    return render_template('how-it-works.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please train models first.'
        }), 500
    
    try:
        # Get the news text from the request
        data = request.json
        if data is None:
            return jsonify({'error': 'No JSON data received'}), 400
            
        news_text = data.get('text', '')
        
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Print received text for debugging
        print(f"Received text for analysis: {news_text[:100]}...")
        
        # Preprocess the text
        processed_text = wordopt(news_text)
        
        # Create a DataFrame with the text
        test_data = pd.DataFrame({'text': [processed_text]})
        
        # Vectorize the text
        print("Vectorizing text...")
        vectorized_text = vectorizer.transform(test_data['text'])
        
        # Get predictions from all models
        print("Getting predictions...")
        
        # Convert numpy values to Python native types to ensure JSON serialization
        results = {}
        
        # Logistic Regression prediction
        lr_pred_class = int(LR.predict(vectorized_text)[0])
        lr_pred_proba = LR.predict_proba(vectorized_text)[0].tolist()
        lr_confidence = lr_pred_proba[1] if lr_pred_class == 1 else lr_pred_proba[0]
        results['lr'] = {
            'prediction': "Real News" if lr_pred_class == 1 else "Fake News",
            'confidence': float(lr_confidence),
            'is_fake': lr_pred_class == 0
        }
        
        # Decision Tree prediction
        dt_pred_class = int(DT.predict(vectorized_text)[0])
        dt_pred_proba = DT.predict_proba(vectorized_text)[0].tolist()
        dt_confidence = dt_pred_proba[1] if dt_pred_class == 1 else dt_pred_proba[0]
        results['dt'] = {
            'prediction': "Real News" if dt_pred_class == 1 else "Fake News",
            'confidence': float(dt_confidence),
            'is_fake': dt_pred_class == 0
        }
        
        # Gradient Boosting prediction
        gb_pred_class = int(GB.predict(vectorized_text)[0])
        gb_pred_proba = GB.predict_proba(vectorized_text)[0].tolist()
        gb_confidence = gb_pred_proba[1] if gb_pred_class == 1 else gb_pred_proba[0]
        results['gb'] = {
            'prediction': "Real News" if gb_pred_class == 1 else "Fake News",
            'confidence': float(gb_confidence),
            'is_fake': gb_pred_class == 0
        }
        
        # Random Forest prediction
        rf_pred_class = int(RF.predict(vectorized_text)[0])
        rf_pred_proba = RF.predict_proba(vectorized_text)[0].tolist()
        rf_confidence = rf_pred_proba[1] if rf_pred_class == 1 else rf_pred_proba[0]
        results['rf'] = {
            'prediction': "Real News" if rf_pred_class == 1 else "Fake News",
            'confidence': float(rf_confidence),
            'is_fake': rf_pred_class == 0
        }
        
        print("Predictions complete, returning results...")
        return jsonify(results)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in prediction route: {e}")
        print(f"Detailed traceback: {error_details}")
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=False)