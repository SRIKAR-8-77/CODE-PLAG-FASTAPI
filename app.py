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
import logging
from typing import Optional, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

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
            return {"original": sample, "gemini": sample, "chatgpt": sample, "claude": sample}

        # Create Crew Task objects and run agents
        task_original = create_code_generation_task(question, language, agent=code_generator_agent)
        task_chatgpt = create_code_generation_task(question, language, agent=chatgpt_generator_agent)
        task_claude = create_code_generation_task(question, language, agent=claude_generator_agent)

        def run_agent(agent, task):
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            out = crew.kickoff()
            return self._strip_code_fence(str(out), language)

        original = run_agent(code_generator_agent, task_original)
        chatgpt = run_agent(chatgpt_generator_agent, task_chatgpt)
        claude = run_agent(claude_generator_agent, task_claude)

        # Return both 'original' and 'gemini' keys for compatibility (gemini == original)
        return {"original": original, "gemini": original, "chatgpt": chatgpt, "claude": claude}

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

    def check_plagiarism(self, user_code: str, question: str, generated_code_original: str, generated_code_chatgpt: str, generated_code_claude: str, language: str) -> dict:
        """Return similar-line lists for each comparison.

        If MOCK_MODE -> use local matcher. Otherwise delegate to the plagiarism_detector_agent (Crew task).
        """
        logger.info("Checking plagiarism: language=%s, mock=%s", language, MOCK_MODE)

        if MOCK_MODE:
            return {
                "gemini_vs_user": self._local_find_similar_lines(user_code, generated_code_original),
                "chatgpt_vs_user": self._local_find_similar_lines(user_code, generated_code_chatgpt),
                "claude_vs_user": self._local_find_similar_lines(user_code, generated_code_claude),
            }

        # Build and run the agent task
        task = create_plagiarism_detection_task(
            user_code=user_code,
            generated_code_original=generated_code_original or "",
            generated_code_chatgpt=generated_code_chatgpt or "",
            generated_code_claude=generated_code_claude or "",
            question=question,
            language=language,
        )
        crew = Crew(agents=[plagiarism_detector_agent], tasks=[task], verbose=False)
        out = crew.kickoff()
        out_str = str(out)

        # Try to extract JSON from agent output
        json_start = out_str.find("{")
        json_end = out_str.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            try:
                parsed = json.loads(out_str[json_start:json_end])
                return parsed
            except Exception:
                logger.exception("Failed to parse agent JSON output")
                return {"error": "failed_to_parse_agent_output", "raw_output": out_str}

        # Fallback: return raw agent output
        return {"error": "no_json_in_agent_output", "raw_output": out_str}


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
                "original": req.gemini_code or "",
                "chatgpt": req.chatgpt_code or "",
                "claude": req.claude_code or "",
            }
        else:
            generated = system.generate_code_solution(req.question, req.language)

        similarity = system.check_plagiarism(
            user_code=req.user_code,
            question=req.question,
            generated_code_original=generated.get("original"),
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

        # If the client supplied any generated codes, use those values (avoid regenerating all three).
        # Only call the LLM generation when none of the codes were supplied.
        supplied_any = any([
            bool(request.gemini_code),
            bool(request.chatgpt_code),
            bool(request.claude_code),
        ])

        if supplied_any:
            generated_codes = {
                "original": request.gemini_code or "",
                "chatgpt": request.chatgpt_code or "",
                "claude": request.claude_code or "",
            }
        else:
            generated_codes = plagiarism_system.generate_code_solution(request.question, request.language)

        similarity = plagiarism_system.check_plagiarism(
            user_code=request.user_code,
            question=request.question,
            generated_code_original=generated_codes.get("original") or generated_codes.get("gemini"),
            generated_code_chatgpt=generated_codes.get("chatgpt"),
            generated_code_claude=generated_codes.get("claude"),
            language=request.language,
        )

        response = {
            "question": request.question,
            "language": request.language,
            "user_code": request.user_code,
            "generated_codes": generated_codes,
            "similar_lines": similarity,
        }

        return JSONResponse(content=response)
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})