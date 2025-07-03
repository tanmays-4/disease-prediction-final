def test_model():
    try:
        # Try to load the model using pickle
        with open('lung_cancer_model.pkl', 'rb') as f:
            import pickle
            model = pickle.load(f)
            
        print("Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        
        # Print model attributes
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                try:
                    print(f"{attr}: {getattr(model, attr)}")
                except:
                    print(f"{attr}: [Not printable]")
        
        # Try to make a prediction
        try:
            # Features: [smoking, yellow_fingers, age, gender]
            test_features = [1, 1, 65, 1]  # Smoker, yellow fingers, 65 years, male
            prediction = model.predict([test_features])
            print(f"\nPrediction for {test_features}: {prediction}")
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([test_features])
                print(f"Probabilities: {proba}")
                
        except Exception as e:
            print(f"\nPrediction error: {str(e)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
