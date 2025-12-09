import requests
import json
import sys

# --- Configuration ---
API_URL = "https://code-plag-fastapi.onrender.com"
TEST_QUESTION = "Write a function to compute the Nth Fibonacci number iteratively."
TEST_LANGUAGE = "C++"

# Sample user code (intentionally uses a standard iterative loop, which should trigger matches)
USER_CODE = """
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    int a = 0;
    int b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}
"""

def call_generate_endpoint(question: str, language: str):
    """Calls the /generate endpoint to get AI code variants and saves request/response objects."""
    print(f"\n--- STEP 1: Calling /generate for {language} ---")
    payload = {"question": question, "language": language}
    
    # 1. SAVE INPUT
    try:
        with open("generate_input.json", "w") as f:
            json.dump(payload, f, indent=4)
        print("  > Saved request input to generate_input.json")
    except IOError as e:
        print(f"  ✗ WARNING: Could not save generate_input.json. Error: {e}", file=sys.stderr)


    try:
        response = requests.post(f"{API_URL}/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # 2. SAVE OUTPUT
        try:
            with open("generate_output.json", "w") as f:
                json.dump(data, f, indent=4)
            print("  > Saved response output to generate_output.json")
        except IOError as e:
             print(f"  ✗ WARNING: Could not save generate_output.json. Error: {e}", file=sys.stderr)

        print("✓ Generation successful.")
        
        # Log generated code snippets (first few lines)
        for key, code in data['generated_codes'].items():
            print(f"  > {key.upper()} Code starts with: {''.join(code.splitlines()[:1])}...")
            
        return data['generated_codes']
    
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR in /generate request: {e}", file=sys.stderr)
        return None

def call_analyze_endpoint(question: str, language: str, user_code: str, generated_codes: dict):
    """Calls the /analyze endpoint to check for plagiarism and saves request/response objects."""
    print(f"\n--- STEP 2: Calling /analyze for Plagiarism Check ---")
    payload = {
        "question": question,
        "language": language,
        "user_code": user_code,
        "gemini_code": generated_codes.get('gemini', ''),
        "chatgpt_code": generated_codes.get('chatgpt', ''),
        "claude_code": generated_codes.get('claude', ''),
    }

    # 3. SAVE INPUT
    try:
        with open("analyze_input.json", "w") as f:
            json.dump(payload, f, indent=4)
        print("  > Saved request input to analyze_input.json")
    except IOError as e:
        print(f"  ✗ WARNING: Could not save analyze_input.json. Error: {e}", file=sys.stderr)

    try:
        response = requests.post(f"{API_URL}/analyze", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        # 4. SAVE OUTPUT
        try:
            with open("analyze_output.json", "w") as f:
                json.dump(data, f, indent=4)
            print("  > Saved response output to analyze_output.json")
        except IOError as e:
             print(f"  ✗ WARNING: Could not save analyze_output.json. Error: {e}", file=sys.stderr)

        print("✓ Analysis successful.")
        return data['similar_lines']
    
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR in /analyze request: {e}", file=sys.stderr)
        return None

def print_analysis_report(similar_lines: dict):
    """Prints the structured report of similar lines."""
    print("\n--- FINAL PLAGIARISM REPORT ---")
    
    for comparison, matches in similar_lines.items():
        title = comparison.upper().replace('_', ' VS ')
        print(f"\n**{title}**")
        
        if matches:
            print(f"  FOUND {len(matches)} SIMILAR LINES:")
            for match in matches:
                # Truncate content for clean output
                content = match['line_content'].strip()
                print(f"    - User Line {match['user_line_number']} matches AI Line {match['ai_line_number']}: '{content}'")
        else:
            print("  NO SIMILAR LINES FOUND.")

def run_full_test():
    """Executes the full end-to-end test."""
    print(f"*** Starting End-to-End Test (Language: {TEST_LANGUAGE}) ***")
    
    # Step 1
    generated_codes = call_generate_endpoint(TEST_QUESTION, TEST_LANGUAGE)
    if not generated_codes:
        print("\nTEST FAILED at Step 1: Could not generate codes.")
        return

    # Step 2
    similar_lines = call_analyze_endpoint(TEST_QUESTION, TEST_LANGUAGE, USER_CODE, generated_codes)
    if not similar_lines:
        print("\nTEST FAILED at Step 2: Could not perform analysis.")
        return

    # Step 3
    print_analysis_report(similar_lines)
    print("\n*** TEST COMPLETE ***")

if __name__ == "__main__":
    run_full_test()