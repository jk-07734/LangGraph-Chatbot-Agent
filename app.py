from fastapi import FastAPI  # for creating the web application
from pydantic import BaseModel  # for structured data data models
from typing import List  # type hint for type annotations
from langchain_tavily import TavilySearch  # tool for handling search results from Tavily
import os  # environment variables handling
from langgraph.prebuilt import create_react_agent  # Function to create a ReAct agent
from langchain_groq import ChatGroq  # for interacting with LLMs
from dotenv import load_dotenv

load_dotenv()
# Retrieve and set API keys for external tools and services
groq_api_key = os.getenv("API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Predefined list of supported model names
MODEL_NAMES = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "deepseek-r1-distill-llama-70b"
]

# Initialise the TavilySearch tool with a specified maximum number of results.
tool_tavily = TavilySearch(max_results=2)  # Allows retrieving up to 2 results

# Combine the TavilySearch and ExecPython tools into a list.
tools = [tool_tavily, ]

# FastAPI application setup with a title
app = FastAPI(title='LangGraph AI Agent')

# Define the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    model_name: str  # Name of the model to use for processing the request
    system_prompt: str  # System prompt for initialising the model
    messages: List[str]  # List of messages in the chat

# Define an endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    if request.model_name not in MODEL_NAMES:
        return {"error": "Invalid model name. Please select a valid model."}
    
    # Initialise the LLM with the selected model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=request.model_name)

    # Create a ReAct agent using the selected LLM and tools
    agent = create_react_agent(llm, tools=tools)

    system_message = [{"role": "system", "content": request.system_prompt}]
    user_messages = [{"role": "user", "content": msg} for msg in request.messages]
    formatted_messages = system_message + user_messages

    state = {"messages": formatted_messages}

    # Process the state using the agent
    result = agent.invoke(state) # can be async or sync

    return result

# Run the application if executed as the main script
if __name__ == '__main__':
    import uvicorn # to run the FastAPI app
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
    # uvicorn.run(app, host='127.0.0.1', port=8000) # Start the app on localhost with port 8000
    