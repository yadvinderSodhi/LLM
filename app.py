from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
# Path to log files
log_file1 = "sample_log_file_1.log"
log_file2 = "sample_log_file_2.log"

app = FastAPI()

# Initialize an empty list to store session history
session_history = []


class Params(BaseModel):
    id:str


@app.post("/process_params")
async def process_params2(params: Params):
    print("Recieved params")
    #print(f"ID: {params.id}")
    logs = search_tool(params.id)
    session_history.append({id:"id","logs":logs})
    #return {"message":f"params received successfully-->{params.id}"}


    print("logs  ------ ", logs)  # Debug line

    return JSONResponse(content={"message": f"logs:\n{logs}"})

# Langchain tool to search logs
def search_toogit l(query):
    logs = search_logs_by_id([log_file1, log_file2], query)
    return logs if logs else ["No records found."]


# Function to search for IDs in log files
def search_logs_by_id(log_files, search_id):
    found_entries = []
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, "r") as file:
                for line in file:
                    if search_id in line:
                        found_entries.append(line)


    return found_entries


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
