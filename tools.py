import crewai_tools
from crewai_tools import tool
import json
from datetime import datetime


@tool
def code_generation_tool(question: str) -> str:
    """
    Helper tool for code generation - stores the question context.
    The actual code generation is done by the Agent using Gemini.
    """
    return f"Code generation task created for: {question}\nTimestamp: {datetime.now().isoformat()}"


@tool
def plagiarism_detection_tool(user_code: str, generated_code: str) -> str:
    """
    Helper tool for plagiarism detection analysis.
    The actual analysis is done by the Agent using Gemini.
    """
    analysis = {
        "analysis_timestamp": datetime.now().isoformat(),
        "user_code_length": len(user_code),
        "generated_code_length": len(generated_code),
        "status": "Analysis tool initialized and ready for agent processing"
    }
    return json.dumps(analysis, indent=2)
