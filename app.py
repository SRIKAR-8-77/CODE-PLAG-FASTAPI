from crewai import Crew
from tasks import (
    create_code_generation_task,
    create_plagiarism_detection_task,
)
from agents import (
    code_generator_agent,
    chatgpt_generator_agent,
    claude_generator_agent,
    plagiarism_detector_agent,
)
import json
import re
import logging # <-- NEW: Import logging module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# --- NEW: Configure basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ------------------------------------

class PlagiarismCheckSystem:
    """System for detecting plagiarism, refactored to be stateless."""
    
    def _clean_code_output(self, raw_output: str, language: str) -> str:
        """Removes markdown formatting and other text from code output, using the specified language."""
        # Use the language variable (lowercased) to match the code block markdown
        language_tag = language.lower()
        # Search for the pattern ```<language>\n...\n```
        code_match = re.search(r"```" + re.escape(language_tag) + r"\n(.*?)\n```", raw_output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: if no markdown, strip and assume it's all code
        return raw_output.strip()

    def generate_code_solution(self, question: str, language: str) -> dict:
        """
        Step 1: Generate code using Gemini in 3 different styles for the specified language.
        """
        print("\n" + "="*60)
        print("STEP 1: CODE GENERATION (3 STYLES)")
        print("="*60)
        print(f"Question: {question} (Language: {language})\n")

        # Define generation and cleaning logic
        def generate_and_clean(agent, task):
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            result = crew.kickoff()
            # Pass language to the cleaning function
            return self._clean_code_output(str(result), language)

        # Create tasks, passing the language
        task_original = create_code_generation_task(question, language, agent=code_generator_agent)
        task_chatgpt = create_code_generation_task(question, language, agent=chatgpt_generator_agent)
        task_claude = create_code_generation_task(question, language, agent=claude_generator_agent)

        # Generate code with Original style
        print("Generating with Original style...")
        generated_code_original = generate_and_clean(code_generator_agent, task_original)
        print("✓ Original style code generated")

        # Generate code with ChatGPT style
        print("Generating with ChatGPT-style...")
        generated_code_chatgpt = generate_and_clean(chatgpt_generator_agent, task_chatgpt)
        print("✓ ChatGPT-style code generated")

        # Generate code with Claude style
        print("Generating with Claude-style...")
        generated_code_claude = generate_and_clean(claude_generator_agent, task_claude)
        print("✓ Claude-style code generated")

        return {
            "gemini": generated_code_original,
            "chatgpt": generated_code_chatgpt,
            "claude": generated_code_claude,
        }
    
    def check_plagiarism(
        self,
        user_code: str,
        question: str,
        generated_code_original: str,
        generated_code_chatgpt: str,
        generated_code_claude: str,
        language: str
    ) -> dict:
        """
        Step 2: Use the plagiarism_detector_agent to detect similar lines.
        """
        print("\n" + "="*60)
        print("STEP 2: PLAGIARISM DETECTION (AGENT BASED)")
        print("="*60)

        # Create a CrewAI task for the plagiarism agent, passing the language
        task = create_plagiarism_detection_task(
            user_code=user_code,
            generated_code_original=generated_code_original,
            generated_code_chatgpt=generated_code_chatgpt,
            generated_code_claude=generated_code_claude,
            question=question,
            language=language
        )
        crew = Crew(agents=[plagiarism_detector_agent], tasks=[task], verbose=False)
        
        print("Agent is analyzing similar lines...")
        agent_output = crew.kickoff()
        agent_result_str = str(agent_output) 
        print("✓ Analysis complete.")

        # Try to parse the agent's JSON output
        try:
            json_start = agent_result_str.find('{')
            json_end = agent_result_str.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = agent_result_str[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise Exception("No JSON object found in agent output")
        except Exception as e:
            print(f"Error parsing agent JSON output: {e}")
            print(f"Raw agent output: {agent_result_str}")
            # Return a structured error
            result = {
                "error": "Failed to parse agent output.",
                "raw_output": agent_result_str,
                "gemini_vs_user": [],
                "chatgpt_vs_user": [],
                "claude_vs_user": []
            }
        
        return result


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for the two-step process ---
class GenerateRequest(BaseModel):
    question: str
    language: str # Added language for dynamic generation

class AnalyzeRequest(BaseModel):
    question: str
    user_code: str
    gemini_code: str
    chatgpt_code: str
    claude_code: str
    language: str # Added language for dynamic analysis formatting

# --- Endpoint 1: Code Generation ---
@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generates AI codes based on the question and specified language."""
    # --- NEW: Log incoming request data ---
    logger.info(f"REQUEST /generate: Question='{request.question[:50]}...', Language='{request.language}'")
    
    try:
        plagiarism_system = PlagiarismCheckSystem()
        # Pass language to the system method
        generated_codes = plagiarism_system.generate_code_solution(
            request.question,
            request.language
        )
        
        response_content = {
            "question": request.question,
            "language": request.language,
            "generated_codes": generated_codes
        }

        # --- NEW: Log successful response data ---
        logger.info(f"RESPONSE /generate: Successfully generated codes for {request.language}. Keys: {list(generated_codes.keys())}")

        return JSONResponse(content=response_content)
    except Exception as e:
        # --- UPDATED: Log errors with context ---
        logger.error(f"ERROR /generate: Failed for question '{request.question[:50]}...'. Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Endpoint 2: Plagiarism Analysis (Similar Lines) ---
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyzes user code against AI codes to find similar lines."""
    # --- NEW: Log incoming request data ---
    logger.info(f"REQUEST /analyze: Question='{request.question[:50]}...', Language='{request.language}', User Code Length: {len(request.user_code)}")
    
    try:
        plagiarism_system = PlagiarismCheckSystem()
        
        # Pass language to the system method
        plagiarism_line_report = plagiarism_system.check_plagiarism(
            user_code=request.user_code,
            question=request.question,
            generated_code_original=request.gemini_code,
            generated_code_chatgpt=request.chatgpt_code,
            generated_code_claude=request.claude_code,
            language=request.language
        )
        
        response_content = {
            "question": request.question,
            "language": request.language,
            "user_code": request.user_code,
            "similar_lines": plagiarism_line_report
        }

        # --- NEW: Log successful response data ---
        # Log the number of matches found in the most important comparison (Gemini vs User)
        num_matches = len(plagiarism_line_report.get('gemini_vs_user', []))
        logger.info(f"RESPONSE /analyze: Analysis complete for {request.language}. Found {num_matches} similar lines (Gemini vs User).")

        return JSONResponse(content=response_content)
    except Exception as e:
        # --- UPDATED: Log errors with context ---
        logger.error(f"ERROR /analyze: Failed for question '{request.question[:50]}...'. Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})