from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .router import route
from .retriever import retrieve
from .rag_chain import generate_answer,assemble_facts
from typing import List, Dict, Any, Optional
app = FastAPI()


class RetrieveRequest(BaseModel):
    query: str
    role: str
    k: Optional[int] = 8
    since: Optional[str] = None  # 👈 must be string, not datetime

class QueryRequest(BaseModel):
    question: str
    role: Optional[str] = "Ruby"
    since: Optional[str] = None

@app.get("/")
def root():
    return {"message": "API is running"}

# Add this endpoint (make sure it's before your other endpoints)
@app.get("/roles")
async def list_roles():
    """List all available roles in the system"""
    from .retriever import ROLE_FILTERS  # Import your role definitions
    return {
        "available_roles": list(ROLE_FILTERS.keys()),
        "default_role": "Ruby"
    }

# @app.post("/ask")
# async def ask_endpoint(request: QueryRequest):
#     try:
#         # Route question
#         selected_role = route(request.question, request.role)
        
#         # Retrieve context
#         retrieved = retrieve(
#             query=request.question,
#             role=selected_role,
#             since=request.since
#         )
        
#         # Assemble deterministic facts
#         facts = assemble_facts(selected_role, request.since)
        
#         # Generate answer with Gemini
#         answer = generate_answer(
#             role=selected_role,
#             question=request.question,
#             facts=facts,
#             retrieved_docs=retrieved
#         )
        
#         return {
#             "role": selected_role,
#             "answer": answer,
#             "sources": [doc["id"] for doc in retrieved]
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



import traceback

@app.post("/ask")
async def ask_endpoint(request: QueryRequest):
    try:
        selected_role = route(request.question, request.role)
        print("Selected role:", selected_role)

        retrieved = retrieve(query=request.question, role=selected_role, since=request.since)
        print("Retrieved:", retrieved)

        facts = assemble_facts(selected_role, request.since)
        print("Facts:", facts)

        answer = generate_answer(
            role=selected_role,
            question=request.question,
            facts=facts,
            retrieved_docs=retrieved
        )
        print("Answer:", answer)

        return {"role": selected_role, "answer": answer, "sources": [doc["id"] for doc in retrieved]}
    except Exception as e:
        print("🔥 ERROR in /ask endpoint:", str(e))
        traceback.print_exc()   # full stack trace in terminal
        raise HTTPException(status_code=500, detail=str(e))
