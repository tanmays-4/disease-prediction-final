# Disease Prediction Web Application

A Flask-based web application for predicting various diseases using machine learning models.

## Features
- Predict multiple diseases (Diabetes, Heart Disease, Lung Cancer, Stroke)
- Interactive chatbot for health-related queries
- Responsive design for all devices

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```
5. Add your API keys to the `.env` file
6. Run the application:
   ```bash
   python app.py
   ```

## Environment Variables

Create a `.env` file with the following variables:
```
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_APP=app.py
FLASK_ENV=development
```

## Deployment to Render

1. Push your code to a GitHub repository
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New" and select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: disease-prediction-app
   - Region: Choose the one closest to your users
   - Branch: main (or your preferred branch)
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
6. Add environment variables:
   - `PYTHON_VERSION`: 3.9.0
   - `GEMINI_API_KEY`: Your Gemini API key
   - `FLASK_APP`: app.py
   - `FLASK_ENV`: production
7. Click "Create Web Service"

## Project Structure

```
.
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version specification
├── Procfile               # Process file for Render
├── render.yaml            # Render deployment configuration
├── wsgi.py               # WSGI entry point
├── .env.example          # Example environment variables
├── static/               # Static files (CSS, JS, images)
└── templates/            # HTML templates
```

## License

This project is licensed under the MIT License.
