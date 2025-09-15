import streamlit as st
import requests

# Streamlit App Configuration
st.set_page_config(page_title="LangGraph Agent UI", layout="centered")

# Define API endpoint
# API_URL = "http://127.0.0.1:8000/chat"
API_URL = "https://langgraph-chatbot-agent.onrender.com/chat"

# Predefined models
MODEL_NAMES = [
    "openai/gpt-oss-120b",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-8b-instant"
]

# Streamlit UI Elements
st.title("LangGraph Chatbot Agent")
st.write("Interact with the LangGraph-based agent using this interface.")

given_system_prompt = st.text_area("Define your AI Agent:", height=10, placeholder="You are an analyst")

selected_model = st.selectbox("Select Model:", MODEL_NAMES)

user_input = st.text_area("Enter your query:", height=150, placeholder="Ask anything")

if st.button("Submit"):
    if user_input.strip():
        try:
            # Send the input to the FastAPI backend
            payload = {"messages": [user_input], "model_name": selected_model, 'system_prompt': given_system_prompt}
            response = requests.post(API_URL, json=payload)

            # Display the response
            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    ai_responses = [
                        message.get("content", "")
                        for message in response_data.get("messages", [])
                        if message.get("type") == "ai"
                    ]

                    if ai_responses:
                        st.subheader("Agent Response:")
                        st.markdown(f"**Final Response:** {ai_responses[-1]}")
                        # for i, response_text in enumerate(ai_responses, 1):
                        #     st.markdown(f"**Response {i}:** {response_text}")
                    else:
                        st.warning("No AI response found in the agent output.")
            else:
                st.error(f"Request failed with status code {response.status_code}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message before clicking 'Send Query'.")