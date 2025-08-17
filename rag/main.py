from rag.scripts.api import app  # Import your FastAPI app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag.scripts.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )