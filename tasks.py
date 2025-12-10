from crewai import Task
from agents import (
    code_generator_agent,
    chatgpt_generator_agent,
    claude_generator_agent,
    plagiarism_detector_agent,
)


def create_code_generation_task(question: str, language: str, agent=None) -> Task:
    """Create a Task that requests code generation for `question` in the specified language."""
    description = f"""Generate {language} code for the following programming question:

Question: {question}

Requirements:
- Write a minimalist {language} code solution
- DO NOT include any comments
- Handle edge cases appropriately
"""

    return Task(
        description=description,
        agent=agent,
        expected_output=f"A single block of {language} code. Do not include markdown formatting or any explanatory text.",
    )


def create_plagiarism_detection_task(
    user_code: str,
    generated_code_gemini: str,
    generated_code_chatgpt: str,
    generated_code_claude: str,
    question: str,
    language: str 
) -> Task:
    """
    Create a task for detecting and listing similar lines between user code and AI-generated codes.
    """
    # Define the dynamic language tag (lowercased for markdown block formatting)
    language_tag = language.lower()
    
    return Task(
        description=f"""
Analyze the following code submissions for the programming question: "{question}"

---
User-Submitted Code:
```{language_tag}
{user_code}
```
---
Gemini Generated Code:
```{language_tag}
{generated_code_gemini}
```
---
ChatGPT-style Generated Code:
```{language_tag}
{generated_code_chatgpt}
```
---
Claude-style Generated Code:
```{language_tag}
{generated_code_claude}
```
---

Your task is to perform three separate comparisons and identify all identical or near-identical lines of code.
Do not calculate a percentage. Only list the matching lines.
Lines should be considered similar if they are identical or have only minor whitespace differences.
List line numbers starting from 1 for each code block.

1.  **Compare User Code vs. Gemini Generated Code:**
    List all similar lines. For each match, provide the line number from the user's code and the line number from the Gemini code, along with the code content.

2.  **Compare User Code vs. ChatGPT-style Generated Code:**
    List all similar lines. For each match, provide the line number from the user's code and the line number from the ChatGPT code, along with the code content.

3.  **Compare User Code vs. Claude-style Generated Code:**
    List all similar lines. For each match, provide the line number from the user's code and the line number from the Claude code, along with the code content.

Format your entire output as a single JSON object. The JSON object must have three keys: "gemini_vs_user", "chatgpt_vs_user", and "claude_vs_user".
Each key should contain a list of objects, where each object represents a single line match and has the keys: "user_line_number", "ai_line_number", and "line_content".

If no similarities are found for a comparison, return an empty list for that key.
Example for one match: {{"user_line_number": 5, "ai_line_number": 4, "line_content": "for i in range(n):"}}
""",
        agent=plagiarism_detector_agent,
        expected_output="""STRICT JSON ONLY. No markdown blocks, no fluff, no conversational text.
Return a valid JSON object with exactly these keys: "gemini_vs_user", "chatgpt_vs_user", "claude_vs_user".
Example:
{
  "gemini_vs_user": [{"user_line_number": 1, "ai_line_number": 2, "line_content": "print('hello')"}],
  "chatgpt_vs_user": [],
  "claude_vs_user": []
}"""
    )