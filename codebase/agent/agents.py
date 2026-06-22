"""Agent definitions: two module-level singletons, built once at import time.

financial_agent has access ONLY to get_financial_data - it cannot reach the
vector store even if a user or the model tries to push it that way. This is
the structural guarantee that exact figures always come from the verified
SQL tables, never from model knowledge.

general_agent has access to both tools, for narrative/qualitative questions
that may also need to reference verified figures.
"""

from config import CONFIG
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

from codebase.agent.prompts import SYSTEM_PROMPT_FINANCIAL, SYSTEM_PROMPT_GENERAL
from codebase.agent.tools import get_financial_data, search_annual_report

# _model = ChatGoogleGenerativeAI(model=CONFIG.GEMINI_MODEL)

_model = ChatMistralAI(
    model="mistral-small-latest",   # Mistral Small 4
    mistral_api_key=CONFIG.MISTRAL_API_KEY
)

financial_agent = create_agent(
    model=_model,
    tools=[get_financial_data],
    system_prompt=SYSTEM_PROMPT_FINANCIAL,
)

general_agent = create_agent(
    model=_model,
    tools=[get_financial_data, search_annual_report],
    system_prompt=SYSTEM_PROMPT_GENERAL,
)
