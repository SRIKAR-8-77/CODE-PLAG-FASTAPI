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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

class PlagiarismCheckSystem:
    """System for detecting plagiarism in student code submissions"""
    
    def __init__(self):
        self.generated_code_original = None
        self.generated_code_chatgpt = None
        self.generated_code_claude = None
        self.question = None
    
    def _clean_code_output(self, raw_output: str) -> str:
        """Removes markdown formatting and other text from code output."""
        code_match = re.search(r"```python\n(.*?)\n```", raw_output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: if no markdown, strip and assume it's all code
        return raw_output.strip()

    def generate_code_solution(self, question: str) -> dict:
        """
        Step 1: Generate code using Gemini in 3 different styles:
        - Original (general)
        - ChatGPT-style
        - Claude-style
        
        Args:
            question: The programming problem/question
            
        Returns:
            Dictionary with generated code from each style
        """
        print("\n" + "="*60)
        print("STEP 1: CODE GENERATION (3 STYLES)")
        print("="*60)
        print(f"Question: {question}\n")
        
        self.question = question

        # Create tasks
        task_original = create_code_generation_task(question, agent=code_generator_agent)
        task_chatgpt = create_code_generation_task(question, agent=chatgpt_generator_agent)
        task_claude = create_code_generation_task(question, agent=claude_generator_agent)

        # Generate code with Original style
        print("Generating with Original style...")
        crew_original = Crew(agents=[code_generator_agent], tasks=[task_original], verbose=False)
        result_original = crew_original.kickoff()
        self.generated_code_original = self._clean_code_output(str(result_original))
        print("✓ Original style code generated")

        # Generate code with ChatGPT style
        print("Generating with ChatGPT-style...")
        crew_chatgpt = Crew(agents=[chatgpt_generator_agent], tasks=[task_chatgpt], verbose=False)
        result_chatgpt = crew_chatgpt.kickoff()
        self.generated_code_chatgpt = self._clean_code_output(str(result_chatgpt))
        print("✓ ChatGPT-style code generated")

        # Generate code with Claude style
        print("Generating with Claude-style...")
        crew_claude = Crew(agents=[claude_generator_agent], tasks=[task_claude], verbose=False)
        result_claude = crew_claude.kickoff()
        self.generated_code_claude = self._clean_code_output(str(result_claude))
        print("✓ Claude-style code generated")

        return {
            "original": self.generated_code_original,
            "chatgpt": self.generated_code_chatgpt,
            "claude": self.generated_code_claude,
        }
    

    def check_plagiarism(self, user_code: str) -> dict:
        """
        Step 2: Use the plagiarism_detector_agent to detect similar lines
        between user code and all generated codes.
        """
        if not self.generated_code_original or not self.question:
            # Handle case where generation might have failed
            self.generated_code_original = self.generated_code_original or "print('Error: AI code not generated')"
            self.generated_code_chatgpt = self.generated_code_chatgpt or "print('Error: AI code not generated')"
            self.generated_code_claude = self.generated_code_claude or "print('Error: AI code not generated')"
            if not self.question:
                 raise ValueError("Question is missing. Cannot proceed.")

        print("\n" + "="*60)
        print("STEP 2: PLAGIARISM DETECTION (AGENT BASED)")
        print("="*60)

        # Create a CrewAI task for the plagiarism agent
        task = create_plagiarism_detection_task(
            user_code=user_code,
            generated_code_original=self.generated_code_original,
            generated_code_chatgpt=self.generated_code_chatgpt,
            generated_code_claude=self.generated_code_claude,
            question=self.question
        )
        crew = Crew(agents=[plagiarism_detector_agent], tasks=[task], verbose=False)
        
        print("Agent is analyzing similar lines...")
        # FIX 1: crew.kickoff() returns an object. Convert it to a string.
        agent_output = crew.kickoff()
        agent_result_str = str(agent_output) 
        print("✓ Analysis complete.")

        # Try to parse the agent's JSON output
        try:
            # The agent's raw output might be wrapped in markdown or other text
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
                # FIX 2: Ensure raw_output is a string, not the CrewOutput object
                "raw_output": agent_result_str,
                "gemini_vs_user": [],
                "chatgpt_vs_user": [],
                "claude_vs_user": []
            }
        
        return result
    
    def full_analysis(self, question: str, user_code: str) -> dict:
        """
        Complete workflow: Generate code in 3 styles and check plagiarism (line matching)
        """
        # Step 1: Generate codes
        generated_codes = self.generate_code_solution(question)
        
        # Step 2: Check for similar lines
        plagiarism_line_report = self.check_plagiarism(user_code)
        
        # Step 3: Combine all information for the frontend
        final_report = {
            "question": question,
            "user_code": user_code,
            "generated_codes": {
                "gemini": generated_codes.get("original"),
                "chatgpt": generated_codes.get("chatgpt"),
                "claude": generated_codes.get("claude")
            },
            "similar_lines": plagiarism_line_report # This contains the { gemini_vs_user, ... } structure
        }
        
        return final_report


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    question: str
    user_code: str

class GenerateRequest(BaseModel):
    question: str


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generates AI codes (gemini/chatgpt/claude) for the given question."""
    try:
        plagiarism_system = PlagiarismCheckSystem()
        generated = plagiarism_system.generate_code_solution(request.question)
        return JSONResponse(content={
            "question": request.question,
            "generated_codes": generated
        })
    except Exception as e:
        print(f"Error during generation: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        plagiarism_system = PlagiarismCheckSystem()
        result = plagiarism_system.full_analysis(request.question, request.user_code)
        # result now contains { question, user_code, generated_codes, similar_lines }
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})