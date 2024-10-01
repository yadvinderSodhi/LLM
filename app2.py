from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from pydantic import BaseModel

import os

app = FastAPI()

# CORS Middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the summarization pipeline from Hugging Face
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Wrap the Hugging Face pipeline in LangChain's LLM wrapper
llm = HuggingFacePipeline(pipeline=summarizer_pipeline)

# Create a prompt template for LangChain
prompt = PromptTemplate(input_variables=["input_text"],
                        template="Summarize the following log entries:\n{input_text}\n"
                                 "Please provide only the summary without additional information.")

# Create the RunnableSequence with the wrapped LLM and prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Path to log files
log_file1 = "sample_log_file_1.log"
log_file2 = "sample_log_file_2.log"


# Function to search for IDs in log files
def search_logs_by_id(log_files, search_id):
    found_entries = []
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, "r") as file:
                for line in file:
                    if search_id in line:
                        found_entries.append(line.strip())
    return found_entries


# Langchain tool to search logs
def search_tool(query):
    logs = search_logs_by_id([log_file1, log_file2], query)
    return logs if logs else ["No records found."]


class Params(BaseModel):
    id:str

# API endpoint to query logs
@app.post("/process_params")
async def query_agent(params: Params):
    print("Recieved params")
    print(f"EVE ID :{params.id}")

    logs = search_tool(params.id)
    print("logs  ------ ", logs)  # Debug line
    # Prepare the input text for summarization
    input_text = "\n".join(logs)

    # Check if there's enough input text
    if not input_text.strip():
        return JSONResponse(content={"response": "No records found."})

    response = llm_chain.invoke({"input_text": input_text})

    # Print the response before returning it
    print("Generated response:", response)
    return JSONResponse(content={"response": response})  # Ensure to return the actual response




