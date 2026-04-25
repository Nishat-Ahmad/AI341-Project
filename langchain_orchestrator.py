# Fleet-Vision - Autonomous Vehicle Inspection & Dispatch
# LangChain Orchestrator for intelligent vehicle inspection workflow

from fastapi import FastAPI

app = FastAPI(title="Fleet-Vision Orchestrator")

@app.get("/health")
def health():
    return {"status": "healthy"}
