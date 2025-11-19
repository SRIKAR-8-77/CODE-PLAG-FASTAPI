import requests
import json

# Define the base URL
base_url = "http://127.0.0.1:8000"

# 1. Define the programming question and the user's code
QUESTION = "Write a Python function to find the max of two numbers."
USER_CODE = """
def find_max(a, b):
    if a > b:
        return a
    else:
        return b
"""

# --- Step 1: Generate AI Codes ---
def step_1_generate_codes():
    print("="*50)
    print("STEP 1: Calling /generate endpoint...")
    print("="*50)
    
    url = f"{base_url}/generate"
    payload = {"question": QUESTION}

    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("Generation successful!")
            result = response.json()
            # print(json.dumps(result, indent=2))
            return result.get("generated_codes", {})
        else:
            print(f"Generation failed with status code: {response.status_code}")
            print("Response:", response.text)
            return None

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server in Step 1.")
        print("Please make sure your FastAPI server (uvicorn) is running.")
        return None

# --- Step 2: Analyze Plagiarism ---
def step_2_analyze_plagiarism(generated_codes):
    print("\n" + "="*50)
    print("STEP 2: Calling /analyze endpoint with user code and generated codes...")
    print("="*50)
    
    url = f"{base_url}/analyze"
    
    # Construct the full payload for the analysis step
    payload = {
        "question": QUESTION,
        "user_code": USER_CODE,
        "gemini_code": generated_codes.get("gemini", ""),
        "chatgpt_code": generated_codes.get("chatgpt", ""),
        "claude_code": generated_codes.get("claude", "")
    }

    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("Analysis successful! Similar lines report received.")
            print("\n" + "="*50)
            print("FINAL SERVER RESPONSE (SIMILAR LINES):")
            print("="*50)
            
            result = response.json()
            print(json.dumps(result, indent=2))
            
            # Print the specific similar lines for clarity
            print("\n--- Similar Lines (Gemini vs User) ---")
            print(result.get("similar_lines", {}).get("gemini_vs_user", "N/A"))
            
        else:
            print(f"Analysis failed with status code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server in Step 2.")
        print("Please make sure your FastAPI server (uvicorn) is running.")


# --- Run the two-step workflow ---
if __name__ == "__main__":
    ai_codes = step_1_generate_codes()
    
    if ai_codes:
        step_2_analyze_plagiarism(ai_codes)