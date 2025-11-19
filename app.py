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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

class PlagiarismCheckSystem:
    """System for detecting plagiarism, refactored to be stateless."""
    
    def _clean_code_output(self, raw_output: str) -> str:
        """Removes markdown formatting and other text from code output."""
        code_match = re.search(r"```python\n(.*?)\n```", raw_output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: if no markdown, strip and assume it's all code
        return raw_output.strip()

    def generate_code_solution(self, question: str) -> dict:
        """
        Step 1: Generate code using Gemini in 3 different styles.
        """
        print("\n" + "="*60)
        print("STEP 1: CODE GENERATION (3 STYLES)")
        print("="*60)
        print(f"Question: {question}\n")

        # Define generation and cleaning logic
        def generate_and_clean(agent, task):
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            result = crew.kickoff()
            return self._clean_code_output(str(result))

        # Create tasks
        task_original = create_code_generation_task(question, agent=code_generator_agent)
        task_chatgpt = create_code_generation_task(question, agent=chatgpt_generator_agent)
        task_claude = create_code_generation_task(question, agent=claude_generator_agent)

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
        generated_code_claude: str
    ) -> dict:
        """
        Step 2: Use the plagiarism_detector_agent to detect similar lines.
        All AI codes are now passed in as arguments.
        """
        print("\n" + "="*60)
        print("STEP 2: PLAGIARISM DETECTION (AGENT BASED)")
        print("="*60)

        # Create a CrewAI task for the plagiarism agent
        task = create_plagiarism_detection_task(
            user_code=user_code,
            generated_code_original=generated_code_original,
            generated_code_chatgpt=generated_code_chatgpt,
            generated_code_claude=generated_code_claude,
            question=question
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

class AnalyzeRequest(BaseModel):
    question: str
    user_code: str
    gemini_code: str
    chatgpt_code: str
    claude_code: str

# --- Endpoint 1: Code Generation ---
@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generates AI codes based on the question."""
    try:
        plagiarism_system = PlagiarismCheckSystem()
        generated_codes = plagiarism_system.generate_code_solution(request.question)
        
        return JSONResponse(content={
            "question": request.question,
            "generated_codes": generated_codes
        })
    except Exception as e:
        print(f"Error during code generation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Endpoint 2: Plagiarism Analysis (Similar Lines) ---
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyzes user code against AI codes to find similar lines."""
    try:
        plagiarism_system = PlagiarismCheckSystem()
        
        plagiarism_line_report = plagiarism_system.check_plagiarism(
            user_code=request.user_code,
            question=request.question,
            generated_code_original=request.gemini_code,
            generated_code_chatgpt=request.chatgpt_code,
            generated_code_claude=request.claude_code
        )
        
        return JSONResponse(content={
            "question": request.question,
            "user_code": request.user_code,
            "similar_lines": plagiarism_line_report
        })
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})