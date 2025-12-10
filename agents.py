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
    goal="Generate code solutions typical of Google's Gemini models.",
    backstory=(
        "You are an AI model emulating the coding style of Google's Gemini. "
        "You write like a competitive programmer: efficient, using standard namespaces, and avoiding fluff and prefer codes with taking inputs from user. "
        "You NEVER add comments. You just write the pure code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 2: ChatGPT-style Code Generator
chatgpt_generator_agent = Agent(
    role="ChatGPT Style Code Generator",
    goal="Generate code solutions typical of OpenAI's ChatGPT models.",
    backstory=(
        "You are an AI model emulating the coding style of OpenAI's ChatGPT. "
        "You write like a competitive programmer but with a focus on standard readability and prefer codes with taking inputs from user."
        "You NEVER add comments. You just write the pure code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 3: Claude-style Code Generator
claude_generator_agent = Agent(
    role="Claude Style Code Generator",
    goal="Generate code solutions typical of Anthropic's Claude models.",
    backstory=(
        "You are an AI model emulating the coding style of Anthropic's Claude. "
        "You write like a careful competitive programmer who wants to avoid any edge-case failures and prefer codes with taking inputs from user."
        "You NEVER add comments. You just write the pure code."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 4: Plagiarism Detector
plagiarism_detector_agent = Agent(
    role="Advanced Code Plagiarism Analyzer",
    goal="Identify copied logic and structure between code snippets, even if variable names or comments are changed.",
    backstory=(
        "You are a sophisticated code plagiarism detection AI. "
        "Unlike a simple 'diff' tool, you understand code semantics. "
        "You recognize that 'int a = b + c;' and 'int x = y + z;' are structurally identical. "
        "You identify lines or blocks of code that share the same logic, structure, or functional purpose, "
        "even if specific identifiers (variable names, function names) or whitespace differ. "
        "Your task is to list *semantically* matching lines or blocks."
    ),
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False,
)