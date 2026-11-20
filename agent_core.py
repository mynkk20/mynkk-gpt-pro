import os
import math
import requests
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# Tool 1: Calculator
@tool
def calculator(expression : str) -> str:
    """
    Useful for solving math calculations.
    Input must be a valid Python math expression.
    """
    try:
        result = eval(expression, {'math': math, '__builtins__': {}})
        return f'result: {result}'
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 2: Live Weather
@tool
def get_live_weather(city: str) -> str:
    """
    Returns real-time current weather for a given city.
    """
    try:
        url = f"https://wttr.in/{city.strip()}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            temp_c = current['temp_C']
            desc = current['weatherDesc'][0]['value']
            humidity = current['humidity']
            return f"🌤️ Weather in {city.title()}: {temp_c}°C, {desc}, Humidity {humidity}%"
        else:
            return f"⚠️ Failed to get weather for '{city}' (Status Code: {response.status_code})"
    except Exception as e:
        return f"⚠️ Error fetching weather data: {str(e)}"

# Tool 3: Knowledge Base
@tool
def knowledge_base(topic: str) -> str:
    """
    Returns information about AI and tech topics.
    """
    kb = {
        "langchain": "LangChain is a Python framework for building apps powered by LLMs.",
        "machine learning": "Machine Learning is a subset of AI where models learn patterns from data.",
        "neural network": "A Neural Network is a system of nodes inspired by the human brain.",
        "llm": "Large Language Models (LLMs) are trained on massive text data to generate human language.",
        "ai agent": "An AI Agent is an LLM that can reason, use tools, and take actions autonomously.",
    }
    key = topic.lower().strip()
    for k, v in kb.items():
        if k in key or key in k:
            return v
    return f"No info found for '{topic}'."

# Tool 4: Web Search
@tool
def web_search(query: str) -> str:
    """
    Searches the live web for answers to current events and real-time facts.
    """
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No relevant search results found."
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(f"[{i}] Title: {r.get('title')}\nSnippet: {r.get('body')}")
            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"⚠️ Web search error: {str(e)}"

# Agent Setup
react_prompt = PromptTemplate.from_template("""You are a helpful AI assistant. Answer the question using the tools available.

You have access to these tools:
{tools}

Use EXACTLY this format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

def get_agent_executor(api_key):
    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key, temperature=0)
    tools = [calculator, get_live_weather, knowledge_base, web_search]
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6, handle_parsing_errors=True)
