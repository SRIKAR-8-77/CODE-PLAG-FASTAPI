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
"""Cleaned FastAPI app for code generation and plagiarism line detection.

Endpoints:
- POST /generate -> { question, language } -> { generated_codes }
- POST /analyze   -> { question, language, user_code, gemini_code?, chatgpt_code?, claude_code? }

Behavior:
- If environment variable `MOCK_MODE` is set to "1", generation and plagiarism use deterministic local mocks (no LLM calls).
- Otherwise generation uses CrewAI agents and plagiarism delegates to the plagiarism agent task.

This file is intentionally simple and readable to aid debugging and testing.
"""

import os
import re
import json
import ast
import logging
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, PlainTextResponse

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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Read mock mode
MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"

app = FastAPI(title="Code Plagiarism API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple health route
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "ok"


# -------------------- Pydantic models --------------------
class GenerateRequest(BaseModel):
    question: str
    language: str


class AnalyzeRequest(BaseModel):
    question: str
    language: str
    user_code: str
    gemini_code: Optional[str] = None
    chatgpt_code: Optional[str] = None
    claude_code: Optional[str] = None


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON object from text that may contain markdown or other noise.
    """
    # 1. Search for the start of the JSON object by looking for one of the specific keys we expect.
    match = re.search(r"\{\s*[\"']gemini_vs_user[\"']", text)
    if not match:
        # Fallback: look for just a brace if generic
        match = re.search(r"\{", text)

    if match:
        json_start = match.start()
        
        # Attempt 1: Standard JSON decoder (raw_decode handles trailing text)
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(text, idx=json_start)
            return parsed
        except json.JSONDecodeError:
            pass

        # Attempt 2: ast.literal_eval (for single quotes or Python-like dicts)
        # We need a rough end point. Last closing brace is a good guess.
        json_end = text.rfind("}") + 1
        if json_end > json_start:
            candidate = text[json_start:json_end]
            try:
                parsed = ast.literal_eval(candidate)
                return parsed
            except Exception:
                pass
    
    return None

# -------------------- Helper system class --------------------
class PlagiarismCheckSystem:
    """System to generate code and detect similar lines."""

    def _strip_code_fence(self, raw: str, language: str) -> str:
        """Strip markdown code fences for the specified language if present."""
        if not raw:
            return ""
        lang = language.lower()
        pattern = rf"```{re.escape(lang)}\n(.*?)\n```"
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback to generic fence if specific language fence is missing
        match_generic = re.search(r"```\n(.*?)\n```", raw, re.DOTALL)
        if match_generic:
             return match_generic.group(1).strip()
        return raw.strip()

    def generate_code_solution(self, question: str, language: str) -> Dict[str, str]:
        """Generate code in three styles. In MOCK_MODE returns deterministic samples."""
        logger.info("Generating code: language=%s", language)

        if MOCK_MODE:
            # Deterministic simple samples for testing
            sample_py = (
                "def example(n):\n"
                "    # mock implementation\n"
                "    return n\n"
            )
            sample_cpp = (
                "int example(int n) {\n    return n;\n}\n"
            )
            sample = sample_py if language.lower().startswith("py") else sample_cpp
            return {"gemini": sample, "chatgpt": sample, "claude": sample}

        # Create Crew Task objects and run agents
        task_gemini = create_code_generation_task(question, language, agent=code_generator_agent)
        task_chatgpt = create_code_generation_task(question, language, agent=chatgpt_generator_agent)
        task_claude = create_code_generation_task(question, language, agent=claude_generator_agent)

        def run_agent(agent, task):
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            out = crew.kickoff()
            return self._strip_code_fence(str(out), language)

        gemini = run_agent(code_generator_agent, task_gemini)
        chatgpt = run_agent(chatgpt_generator_agent, task_chatgpt)
        claude = run_agent(claude_generator_agent, task_claude)

        return {"gemini": gemini, "chatgpt": chatgpt, "claude": claude}

    def _local_find_similar_lines(self, a: str, b: str) -> list:
        """Simple exact-line matching (whitespace normalized) used in MOCK_MODE."""
        lines_a = [ln.strip() for ln in a.splitlines() if ln.strip()]
        lines_b = [ln.strip() for ln in b.splitlines() if ln.strip()]
        matches = []
        for i, la in enumerate(lines_a, start=1):
            for j, lb in enumerate(lines_b, start=1):
                if la == lb:
                    matches.append({"user_line_number": i, "ai_line_number": j, "line_content": la})
        return matches

    def check_plagiarism(self, user_code: str, question: str, generated_code_gemini: str, generated_code_chatgpt: str, generated_code_claude: str, language: str) -> dict:
        """Return similar-line lists for each comparison.

        If MOCK_MODE -> use local matcher. Otherwise delegate to the plagiarism_detector_agent (Crew task).
        """
        logger.info("Checking plagiarism: language=%s, mock=%s", language, MOCK_MODE)

        if MOCK_MODE:
            return {
                "gemini_vs_user": self._local_find_similar_lines(user_code, generated_code_gemini),
                "chatgpt_vs_user": self._local_find_similar_lines(user_code, generated_code_chatgpt),
                "claude_vs_user": self._local_find_similar_lines(user_code, generated_code_claude),
            }

        # Build and run the agent task
        task = create_plagiarism_detection_task(
            user_code=user_code,
            generated_code_gemini=generated_code_gemini or "",
            generated_code_chatgpt=generated_code_chatgpt or "",
            generated_code_claude=generated_code_claude or "",
            question=question,
            language=language,
        )
        crew = Crew(agents=[plagiarism_detector_agent], tasks=[task], verbose=False)
        out = crew.kickoff()
        out_str = str(out)

        parsed = extract_json_from_text(out_str)
        if parsed:
            return parsed

        # Fallback/Error if extraction failed
        logger.error("Could not find valid JSON structure in agent output.")
        return {"error": "failed_to_parse_agent_output", "raw_output": out_str}


# -------------------- Endpoints --------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate AI code variants for a question and language."""
    try:
        system = PlagiarismCheckSystem()
        codes = system.generate_code_solution(req.question, req.language)
        return JSONResponse(content={"question": req.question, "language": req.language, "generated_codes": codes})
    except Exception as e:
        logger.exception("/generate failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyze user code vs AI-generated codes. Accepts pre-generated codes or generates them when absent."""
    try:
        system = PlagiarismCheckSystem()

        # If any code supplied, use supplied values (empty -> treated as empty string)
        supplied_any = any([req.gemini_code, req.chatgpt_code, req.claude_code])
        if supplied_any:
            generated = {
                "gemini": req.gemini_code or "",
                "chatgpt": req.chatgpt_code or "",
                "claude": req.claude_code or "",
            }
        else:
            generated = system.generate_code_solution(req.question, req.language)

        similarity = system.check_plagiarism(
            user_code=req.user_code,
            question=req.question,
            generated_code_gemini=generated.get("gemini"),
            generated_code_chatgpt=generated.get("chatgpt"),
            generated_code_claude=generated.get("claude"),
            language=req.language,
        )

        response = {
            "question": req.question,
            "language": req.language,
            "user_code": req.user_code,
            "generated_codes": generated,
            "similar_lines": similarity,
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.exception("/analyze failed")
        return JSONResponse(status_code=500, content={"error": str(e)})