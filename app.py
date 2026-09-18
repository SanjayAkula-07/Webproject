from fastapi import FastAPI
from pydantic import BaseModel
from secure_fin_agent import FinSecureAgent

app = FastAPI()
agent = FinSecureAgent()

class QueryRequest(BaseModel):
    user: dict
    prompt: str
    documents: list = []

@app.post("/query")
def run_query(req: QueryRequest):
    # This takes the incoming web request and feeds it to your agent
    result = agent.execute_query(req.user, req.prompt, req.documents)
    return result