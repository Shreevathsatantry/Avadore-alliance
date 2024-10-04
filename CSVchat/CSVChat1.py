import streamlit as st
from langchain_community.llms import Ollama
import pandas as pd
import os
import uuid
from pandasai import SmartDataframe, Agent
from AutoClean import AutoClean

llm = Ollama(model="qwen2.5:7b-instruct-q4_0", base_url="https://32a1-35-240-171-223.ngrok-free.app/")

if 'history' not in st.session_state:
    st.session_state.history = []

session_id = str(uuid.uuid4())
session_dir = f"content/exports/charts/{session_id}"
os.makedirs(session_dir, exist_ok=True)

st.title("Data Analysis with PandasAI")

uploader_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    pipeline = AutoClean(data)
    cleaned_data = pipeline.output
    st.write(cleaned_data.head(3))

    df = SmartDataframe(cleaned_data, config={"llm": llm})
    # agent = Agent(cleaned_data, config={"llm": llm}, description="the query should reflect the exact names of the columns")

    prompt = st.chat_input("Enter your prompt:")

    if prompt:
        with st.spinner("Generating response..."):
            # rephrase = agent.rephrase_query(prompt)
            response = df.chat(prompt)

            with st.expander("Table:"):
                st.write(response)

            with st.expander("Rephrase:"):
                st.write(prompt)

            # Ensure the response is treated as a string
            response_str = str(response)

            # Check if the response includes a chart file path
            if "/charts/" in response_str:
                # Generate a unique filename for the chart in the session-specific directory
                unique_filename = f"chart_{uuid.uuid4()}.png"
                saved_path = os.path.join(session_dir, unique_filename)

                # Assuming the chart is generated and saved at the given path
                os.rename(response_str.strip(), saved_path)  # Move or rename the chart to the session directory

                # Update the response to include the new path
                response_str = saved_path

            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.history.append({"role": "assistant", "content": response_str})

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        content = str(message["content"])
        st.write(content)

        if message["role"] == "assistant" and content.endswith(".png"):
            image_path = content
            if os.path.exists(image_path):
                st.image(image_path)
            else:
                st.error("Image file not found!")
