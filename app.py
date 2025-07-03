from flask import Flask, render_template, request, send_from_directory, jsonify, make_response
import pickle
import numpy as np
import os
import requests
from flask_cors import CORS
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set. Chat functionality will not work.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Successfully configured Gemini API")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure CORS
cors = CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/static/*": {"origins": "*"}
})

# Load the models and scalers
try:
    # Load diabetes model and scaler
    diabetes_model_path = os.path.join(os.path.dirname(__file__), 'classifier.pkl')
    diabetes_scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    
    # Load heart disease model
    heart_model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model (1).pkl')
    heart_scaler_path = os.path.join(os.path.dirname(__file__), 'heart_scaler.pkl')
    
    # Load lung cancer model
    lung_model_path = os.path.join(os.path.dirname(__file__), 'lung_cancer_model.pkl')
    
    # Load diabetes model and scaler
    diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
    diabetes_scaler = pickle.load(open(diabetes_scaler_path, 'rb'))
    
    # Load heart disease model and scaler
    heart_model = pickle.load(open(heart_model_path, 'rb'))
    heart_scaler = pickle.load(open(heart_scaler_path, 'rb')) if os.path.exists(heart_scaler_path) else None
    
    # Load lung cancer model
    lung_model = pickle.load(open(lung_model_path, 'rb'))
    
    # Load stroke model (create a placeholder if not exists)
    stroke_model_path = os.path.join(os.path.dirname(__file__), 'stroke_model.pkl')
    stroke_model = None
    if os.path.exists(stroke_model_path):
        stroke_model = pickle.load(open(stroke_model_path, 'rb'))
    else:
        print("Warning: stroke_model.pkl not found. Creating placeholder model...")
        # Create a simple placeholder model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create dummy training data with stroke features
        np.random.seed(42)
        X_dummy = np.random.randn(1000, 10)  # 10 features for stroke prediction
        y_dummy = np.random.randint(0, 2, 1000)  # Binary classification
        
        stroke_model = RandomForestClassifier(n_estimators=100, random_state=42)
        stroke_model.fit(X_dummy, y_dummy)
        
        # Save the placeholder model
        pickle.dump(stroke_model, open(stroke_model_path, 'wb'))
        print("Placeholder stroke model created and saved.")
    
    print("All models and scalers loaded successfully!")
    
    # Debug: Print model information
    if lung_model is not None:
        print("\nLung Cancer Model Info:")
        print(f"Model type: {type(lung_model).__name__}")
        if hasattr(lung_model, 'n_features_in_'):
            print(f"Expected number of features: {lung_model.n_features_in_}")
        if hasattr(lung_model, 'feature_importances_'):
            print(f"Number of feature importances: {len(lung_model.feature_importances_)}")
        print(f"Model parameters: {lung_model.get_params()}")
        
except Exception as e:
    print(f"Error loading models or scalers: {str(e)}")
    diabetes_model = diabetes_scaler = heart_model = heart_scaler = lung_model = None

@app.route('/test_lung_model')
def test_lung_model():
    """Test route to debug the lung cancer model"""
    try:
        if lung_model is None:
            return "Lung cancer model not loaded"
            
        # Test cases with different feature combinations
        test_cases = [
            # [smoking, yellow_fingers, age, gender]
            [1, 1, 65, 1],  # High risk: smoker, yellow fingers, older male
            [0, 0, 30, 0],  # Low risk: non-smoker, no yellow fingers, young female
            [1, 0, 50, 1],  # Medium risk: smoker, no yellow fingers, middle-aged male
            [0, 1, 70, 0]   # Medium risk: non-smoker, yellow fingers, older female
        ]
        
        results = []
        for features in test_cases:
            try:
                # Make prediction
                prediction = lung_model.predict([features])[0]
                
                # Get probability if available
                if hasattr(lung_model, 'predict_proba'):
                    proba = lung_model.predict_proba([features])[0][1]
                else:
                    proba = 0.5
                
                results.append({
                    'features': features,
                    'prediction': 'High risk' if prediction == 1 else 'Low risk',
                    'confidence': f"{proba*100:.1f}%"
                })
            except Exception as e:
                results.append({
                    'features': features,
                    'error': str(e)
                })
        
        # Return results as HTML for easy viewing
        html = "<h1>Lung Cancer Model Test</h1>"
        html += "<table border='1'><tr><th>Features</th><th>Prediction</th><th>Confidence</th></tr>"
        for result in results:
            html += "<tr>"
            html += f"<td>{result['features']}</td>"
            if 'error' in result:
                html += f"<td colspan='2' style='color:red'>{result['error']}</td>"
            else:
                html += f"<td>{result['prediction']}</td>"
                html += f"<td>{result['confidence']}</td>"
            html += "</tr>"
        html += "</table>"
        
        # Add model info
        html += "<h2>Model Info</h2>"
        html += f"<p>Model type: {type(lung_model).__name__}</p>"
        if hasattr(lung_model, 'n_features_in_'):
            html += f"<p>Expected features: {lung_model.n_features_in_}</p>"
        if hasattr(lung_model, 'classes_'):
            html += f"<p>Classes: {lung_model.classes_}</p>"
        
        return html
        
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_with_gemini():
    try:
        print("Received chat request")
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        user_message = data.get('message', '').strip()
        print(f"User message: {user_message}")
        
        if not user_message:
            print("Empty message received")
            return jsonify({'error': 'Empty message'}), 400
            
        if not GEMINI_API_KEY:
            print("Gemini API key not configured")
            return jsonify({
                'error': 'Chat functionality is not available. Please contact support.'
            }), 503
            
        try:
            # Initialize the Gemini Flash 2.5 model
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 1024,
                    'top_p': 0.9,
                    'top_k': 40
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            )
            
            # Create chat history
            chat = model.start_chat(history=[])
            
            # Generate response with context
            prompt = f"""You are a helpful health assistant. Provide clear, accurate, and concise responses about health topics. 
            If you don't know something, say so. Never provide medical advice, only general information.
            
            User: {user_message}"""
            
            response = chat.send_message(prompt)
            
            print(f"Response generated: {response.text[:200]}...")
            return jsonify({
                'response': response.text
            })
            
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            return jsonify({
                'error': f'Error processing your request: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/lung')
def lung():
    return render_template('lung.html')

@app.route('/stroke')
def stroke():
    return render_template('stroke.html')

@app.route('/predict/lung', methods=['POST'])
def predict_lung():
    try:
        if lung_model is None:
            return render_template('lung.html', 
                               result="Lung cancer prediction model not available. Please try again later.",
                               error=True)
        
        print("\nForm data received:", request.form)  # Debug print
        
        # Get and validate form data
        try:
            age = int(request.form.get('age', 0))
            smokes = int(request.form.get('smokes', 0))
            area_q = int(request.form.get('areaQ', 0))
            alkhol = int(request.form.get('alkhol', 0))
            
            # Validate age
            if not (0 <= age <= 120):
                raise ValueError("Age must be between 0 and 120")
            
            # Prepare features for prediction
            features = [age, smokes, area_q, alkhol]
            
            print("\nProcessed features:", features)  # Debug print
            
            # Make prediction
            prediction = lung_model.predict([features])[0]
            
            # Get probability if available, otherwise use a default confidence
            if hasattr(lung_model, 'predict_proba'):
                probability = lung_model.predict_proba([features])[0][1]
            else:
                # If model doesn't support probabilities, use a default confidence
                probability = 0.85 if prediction == 1 else 0.15
            
            # Format the result
            result = "High risk of lung cancer" if prediction == 1 else "Low risk of lung cancer"
            confidence = round(probability * 100, 2)
            
            print(f"\nPrediction: {result} (Confidence: {confidence}%)")  # Debug print
            
            form_data = {
                'age': request.form.get('age', ''),
                'smokes': request.form.get('smokes', ''),
                'areaQ': request.form.get('areaQ', ''),
                'alkhol': request.form.get('alkhol', '')
            }
            
            return render_template('lung.html', 
                               result=result,
                               confidence=confidence,
                               show_result=True,
                               form_data=form_data)
            
        except ValueError as ve:
            print(f"ValueError: {str(ve)}")
            return render_template('lung.html',
                               result=f"Invalid input: {str(ve)}",
                               error=True)
            
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(f"Error in lung cancer prediction: {error_msg}")
        import traceback
        traceback.print_exc()
        return render_template('lung.html', 
                           result=error_msg,
                           error=True)

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if diabetes_model is None or diabetes_scaler is None:
        return render_template('diabetes.html', 
                           result="Diabetes model or scaler not loaded. Prediction unavailable.",
                           error=True)
    
    if request.method == 'POST':
        try:
            # Debug: Print raw form data
            print("\nDiabetes prediction form data:", request.form)
            
            # Get form data and convert to float
            data = [float(x) for x in request.form.values()]
            print("Converted data:", data)
            
            # Convert to numpy array and reshape for scaling
            input_data = np.array(data).reshape(1, -1)
            print("Input data shape:", input_data.shape)
            print("Input data values:", input_data)
            
            # Scale the input data
            scaled_data = diabetes_scaler.transform(input_data)
            
            # Make prediction
            prediction = diabetes_model.predict(scaled_data)[0]
            print("Raw prediction:", prediction)
            
            # Get prediction probability if available
            try:
                probabilities = diabetes_model.predict_proba(scaled_data)[0]
                print("Class probabilities:", probabilities)
                probability = probabilities[1]  # Probability of class 1 (Diabetic)
                result = f"Result: {'Diabetic' if prediction == 1 else 'Not Diabetic'} " \
                        f"(Confidence: {probability:.2%})"
            except AttributeError:
                result = f"Result: {'Diabetic' if prediction == 1 else 'Not Diabetic'}"
                
            return render_template('diabetes.html', result=result)
            
        except ValueError as ve:
            print(f"ValueError: {str(ve)}")
            return render_template('diabetes.html', 
                                 result=f"Invalid input: {str(ve)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('diabetes.html', 
                                 result=f"Prediction error: {str(e)}")

@app.route('/predict/stroke', methods=['POST'])
def predict_stroke():
    if stroke_model is None:
        return render_template('stroke.html', 
                           result="Stroke prediction model not loaded. Prediction unavailable.",
                           error=True)
    
    if request.method == 'POST':
        try:
            print("\nStroke prediction form data:", request.form)
            
            # Get form data and convert to appropriate types
            gender = request.form.get('gender', 'Male')
            age = float(request.form.get('age', 0))
            hypertension = int(request.form.get('hypertension', 0))
            heart_disease = int(request.form.get('heart_disease', 0))
            ever_married = request.form.get('ever_married', 'Yes')
            work_type = request.form.get('work_type', 'Private')
            residence_type = request.form.get('Residence_type', 'Urban')
            avg_glucose_level = float(request.form.get('avg_glucose_level', 0))
            bmi = float(request.form.get('bmi', 0))
            smoking_status = request.form.get('smoking_status', 'never smoked')
            
            # Validate inputs
            if age <= 0 or avg_glucose_level <= 0 or bmi <= 0:
                raise ValueError("Please provide valid positive values for all fields")
            
            # Initialize all features to 0
            features = [0.0] * 16  # Initialize with 16 zeros
            
            # Numerical features (direct mapping)
            features[0] = age
            features[1] = float(hypertension)
            features[2] = float(heart_disease)
            features[3] = avg_glucose_level
            features[4] = bmi
            
            # One-hot encode categorical features
            # Gender (features[5:8])
            if gender.lower() == 'male':
                features[5] = 1.0  # gender_Male
            elif gender.lower() == 'other':
                features[6] = 1.0  # gender_Other
            # Else gender_Female is implied by all zeros
            
            # Ever married (feature[7])
            if ever_married.lower() == 'yes':
                features[7] = 1.0  # ever_married_Yes
            
            # Work type (features[8:12])
            work_type_lower = work_type.lower().replace(' ', '_')
            if work_type_lower == 'never_worked':
                features[8] = 1.0  # work_type_Never_worked
            elif work_type_lower == 'private':
                features[9] = 1.0  # work_type_Private
            elif work_type_lower == 'self-employed':
                features[10] = 1.0  # work_type_Self-employed
            elif work_type_lower == 'children':
                features[11] = 1.0  # work_type_children
            # Else work_type_Govt_job is implied by all zeros
            
            # Residence type (feature[12])
            if residence_type.lower() == 'urban':
                features[12] = 1.0  # Residence_type_Urban
            
            # Smoking status (features[13:16])
            if smoking_status.lower() == 'formerly smoked':
                features[13] = 1.0  # smoking_status_formerly smoked
            elif smoking_status.lower() == 'never smoked':
                features[14] = 1.0  # smoking_status_never smoked
            elif smoking_status.lower() == 'smokes':
                features[15] = 1.0  # smoking_status_smokes
            # Else smoking_status_unknown is implied by all zeros
            
            print(f"Processed features: {features}")
            print(f"Number of features: {len(features)}")
            
            # Make prediction with the properly formatted feature vector
            prediction = stroke_model.predict([features])[0]
            
            # Get probability if available
            if hasattr(stroke_model, 'predict_proba'):
                probability = stroke_model.predict_proba([features])[0][1]
            else:
                # Fallback risk calculation if predict_proba is not available
                risk_score = 0.0
                # Age is a major risk factor
                if age > 60: risk_score += 0.3
                elif age > 40: risk_score += 0.15
                # Medical conditions
                if hypertension: risk_score += 0.2
                if heart_disease: risk_score += 0.2
                # Glucose levels
                if avg_glucose_level > 200: risk_score += 0.2
                elif avg_glucose_level > 140: risk_score += 0.1
                # BMI
                if bmi > 35: risk_score += 0.15
                elif bmi > 30: risk_score += 0.1
                # Smoking
                if smoking_status == 'smokes': risk_score += 0.15
                elif smoking_status == 'formerly smoked': risk_score += 0.05
                
                probability = min(risk_score, 0.95)  # Cap at 95%
            
            # Format the result with detailed explanation
            risk_level = "High"
            recommendation = "Please consult a healthcare professional immediately for a comprehensive evaluation."
            
            if probability > 0.6:
                risk_level = "High"
                recommendation = "We strongly recommend scheduling a consultation with a healthcare provider as soon as possible."
            elif probability > 0.3:
                risk_level = "Moderate"
                recommendation = "Consider making lifestyle changes and consult a healthcare provider for a check-up."
            else:
                risk_level = "Low"
                recommendation = "Maintain a healthy lifestyle and regular check-ups."
            
            result = f"{risk_level} Risk of Stroke (Risk Score: {probability:.1%})"
            
            print(f"Prediction result: {result}")
            
            return render_template('stroke.html', 
                               result=result,
                               recommendation=recommendation,
                               show_result=True,
                               form_data=request.form)
            
        except ValueError as ve:
            return render_template('stroke.html', 
                               result=f"Validation Error: {str(ve)}",
                               error=True,
                               form_data=request.form)
        except Exception as e:
            print(f"Error during stroke prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('stroke.html', 
                               result="An error occurred during prediction. Please try again.",
                               error=True,
                               form_data=request.form)

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    if heart_model is None:
        return render_template('heart.html', 
                           result="Heart disease model not loaded. Prediction unavailable.",
                           error=True)
    
    if request.method == 'POST':
        try:
            print("\nHeart disease prediction form data:", request.form)
            
            # Get form data and convert to appropriate types
            age = int(request.form.get('age', 0))
            sex = 1 if request.form.get('sex') == 'male' else 0
            cp = int(request.form.get('cp', 0))  # chest pain type
            trestbps = int(request.form.get('trestbps', 0))  # resting blood pressure
            chol = int(request.form.get('chol', 0))  # serum cholesterol
            fbs = 1 if request.form.get('fbs') == 'true' else 0  # fasting blood sugar
            restecg = int(request.form.get('restecg', 0))  # resting electrocardiographic results
            thalach = int(request.form.get('thalach', 0))  # maximum heart rate achieved
            exang = 1 if request.form.get('exang') == 'true' else 0  # exercise induced angina
            oldpeak = float(request.form.get('oldpeak', 0))  # ST depression induced by exercise
            slope = int(request.form.get('slope', 0))  # slope of the peak exercise ST segment
            ca = int(request.form.get('ca', 0))  # number of major vessels (0-3)
            thal = int(request.form.get('thal', 0))  # thalassemia
            
            # Prepare features for prediction
            features = [
                age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal
            ]
            
            # Scale features if scaler is available
            if heart_scaler:
                features = heart_scaler.transform([features])
            
            # Make prediction
            prediction = heart_model.predict([features])[0]
            probability = heart_model.predict_proba([features])[0][1] if hasattr(heart_model, 'predict_proba') else 0.75 if prediction == 1 else 0.25
            
            # Format the result
            result = "Positive" if prediction == 1 else "Negative"
            confidence = probability * 100 if prediction == 1 else (1 - probability) * 100
            
            return render_template('heart.html', 
                               result=result,
                               confidence=f"{confidence:.2f}%",
                               show_result=True)
            
        except Exception as e:
            print(f"Error during heart disease prediction: {str(e)}")
            return render_template('heart.html', 
                               result=f"Error: {str(e)}",
                               error=True)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def create_app():
    # This function is used by Gunicorn/Railway
    # Ensure the static and templates folders exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("\n=== Starting Flask Application ===")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Set port and check if running in production
    port = int(os.environ.get('PORT', 10000))
    
    # Configure for Railway
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        logger.info("Running in Railway production environment")
        app.config['DEBUG'] = False
    else:
        logger.info("Running in development mode")
        app.config['DEBUG'] = True
        
        # Debug information only in development
        logger.info("\n=== Debug Information ===")
        logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
        logger.info(f"Gemini API Key: {'Set' if os.environ.get('GEMINI_API_KEY') else 'Not Set'}")
    
    return app
    
    logger.info("\n=== Application Starting ===\n")
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=not os.environ.get('RENDER'))