#!/usr/bin/env python3
"""
Test script to verify the chatbot API functionality
"""
import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_chatbot_api():
    """Test the /get_guidance endpoint"""
    
    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    # Test data
    test_diseases = ["diabetes", "heart disease", "lung cancer"]
    
    # API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Health Assistant"
    }
    
    print("Testing DeepSeek API directly...")
    print("=" * 50)
    
    for disease in test_diseases:
        print(f"\nTesting guidance for: {disease}")
        print("-" * 30)
        
        # Create prompt
        prompt = f"""
        Provide detailed guidance and health information about {disease}. Include:
        1. Overview of the condition
        2. Common symptoms
        3. Recommended lifestyle changes
        4. Treatment options
        5. When to see a doctor
        6. Prevention tips
        
        Format the response in clear, easy-to-read markdown with proper headings and bullet points.
        """
        
        # Payload
        payload = {
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            print("Making API request...")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print("✅ SUCCESS: API returned valid response")
                    print(f"Response length: {len(content)} characters")
                    print(f"First 200 characters: {content[:200]}...")
                else:
                    print("❌ FAILURE: No choices in response")
                    print(f"Response: {result}")
            else:
                print(f"❌ FAILURE: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ FAILURE: Request timed out")
        except requests.exceptions.RequestException as e:
            print(f"❌ FAILURE: Request error - {e}")
        except Exception as e:
            print(f"❌ FAILURE: Unexpected error - {e}")
        
        time.sleep(2)  # Rate limiting
    
    print("\n" + "=" * 50)
    print("API testing completed!")

if __name__ == "__main__":
    test_chatbot_api()
