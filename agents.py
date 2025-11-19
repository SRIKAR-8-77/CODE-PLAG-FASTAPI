import os
from dotenv import load_dotenv
from crewai import Agent
from crewai.llm import LLM

# Load environment variables from .env
load_dotenv()

# Get API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")

# Single Gemini LLM instance (used for all agents)
gemini_llm = LLM(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

# Agent 1: Code Generator (Original/General Style)
code_generator_agent = Agent(
    role="Minimalist Code Generator",
    goal="generate simple, minimalist python code",
    backstory=(
        "You are a code generator. You write simple, functional Python code. "
        "You NEVER add comments. You just write the code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 2: ChatGPT-style Code Generator
chatgpt_generator_agent = Agent(
    role="Minimalist Code Generator",
    goal="generate simple, minimalist python code",
    backstory=(
        "You are a code generator. You write simple, functional Python code. "
        "You NEVER add comments. You just write the code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 3: Claude-style Code Generator
claude_generator_agent = Agent(
    role="Minimalist Code Generator",
    goal="generate simple, minimalist python code",
    backstory=(
        "You are a code generator. You write simple, functional Python code. "
        "You NEVER add comments. You just write the code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 4: Plagiarism Detector
plagiarism_detector_agent = Agent(
    role="Code Diff Analyzer",
    goal="Perform a line-by-line comparison of two code snippets and extract all matching or highly similar lines",
    backstory=(
        "You are an expert in code analysis, similar to a 'diff' utility. You do not care about overall similarity or style. "
        "Your sole purpose is to read two pieces of code and meticulously identify and list every single line "
        "that is identical or functionally identical. You output this list of matches with their corresponding line numbers."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)