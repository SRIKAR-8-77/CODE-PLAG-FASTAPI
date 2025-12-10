import os
from dotenv import load_dotenv
from crewai import Agent
from crewai.llm import LLM

# Load environment variables from .env
load_dotenv()

# Get API keys from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # NOTE: In a managed environment, this key is often provided automatically, but 
    # checking for it is good practice for local development.
    pass

# Single Gemini LLM instance (used for all agents)
# Assuming a default API key or environment setup handles the key.
gemini_llm = LLM(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)

# Agent 1: Code Generator (Original/General Style)
code_generator_agent = Agent(
    role="Gemini Style Code Generator",
    goal="Generate concise, efficient, and direct code solutions typical of Google's Gemini models.",
    backstory=(
        "You are an AI model emulating the coding style of Google's Gemini. "
        "You prioritize highly optimized, clean, and modern code solutions. "
        "You write like a competitive programmer: efficient, using standard namespaces, and avoiding fluff. "
        "You NEVER add comments. You just write the pure code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 2: ChatGPT-style Code Generator
chatgpt_generator_agent = Agent(
    role="ChatGPT Style Code Generator",
    goal="Generate clear, standard, and idiomatic code solutions typical of OpenAI's ChatGPT models.",
    backstory=(
        "You are an AI model emulating the coding style of OpenAI's ChatGPT. "
        "Your code is readable, structural, and follows standard conventions. "
        "While you can be efficient, you prioritize clarity and standard practices widely used in the industry. "
        "You write like a competitive programmer but with a focus on standard readability. "
        "You NEVER add comments. You just write the pure code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 3: Claude-style Code Generator
claude_generator_agent = Agent(
    role="Claude Style Code Generator",
    goal="Generate safe, robust, and pedantic code solutions typical of Anthropic's Claude models.",
    backstory=(
        "You are an AI model emulating the coding style of Anthropic's Claude. "
        "You are meticulous, prioritizing safety, correctness, and defensive programming. "
        "Your code often includes explicit type handling and careful structure. "
        "You write like a careful competitive programmer who wants to avoid any edge-case failures. "
        "You NEVER add comments. You just write the pure code."
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