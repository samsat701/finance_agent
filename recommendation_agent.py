from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------
# Web Agent: searches for additional recommendation sources
# -----------------------------------------------------------
web_agent = Agent(
    name="Web Agent",
    role="Search the web for stock analyst recommendation sources",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=[
        "Search for the top three websites (excluding Yahoo Finance) that provide credible stock analyst recommendations. " 
        "Include website names and URLs as sources."
    ],
    show_tool_calls=True,
    markdown=True,
    #debug_mode=True,
)

# -----------------------------------------------------------
# Finance Agent: gets financial data and news
# -----------------------------------------------------------
finance_agent = Agent(
    name="Finance Agent",
    role="Gather financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True
        )
    ],
    instructions=[
        "Retrieve the latest stock prices, analyst recommendations, company info, and news. " 
        "Display data in tables where appropriate and always include sources."
    ],
    show_tool_calls=True,
    markdown=True,
    #debug_mode=True,
)

# -----------------------------------------------------------
# Agent Team: combine both agents
# -----------------------------------------------------------
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    instructions=[
        "Use the Web Agent to search for additional sources on stock analyst recommendations (other than Yahoo Finance).",
        "Use the Finance Agent to gather current financial data, including stock prices, analyst recommendations, company info, and news.",
        "Analyze all the gathered information and determine which stock is the best buy right now.",
        "Provide a detailed, step-by-step reasoning with tables (if useful) and include all relevant sources."
    ],
    show_tool_calls=True,
    markdown=True,
    #debug_mode=True,
)

# -----------------------------------------------------------
# Final Answer from Both Agents
# -----------------------------------------------------------
agent_team.print_response(
    "Based on the latest news and financial data, which stock is the best buy right now? "
    "Please provide your detailed reasoning, weigh all sources (including those from alternative websites), "
    "and present your final recommendation with sources.",
    stream=True
)