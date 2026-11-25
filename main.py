import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.answer import Assistant

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "queries.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and index once at startup, not on every request
    LOG_DIR.mkdir(exist_ok=True)
    state["assistant"] = Assistant()
    logger.info("Assistant ready")
    yield


app = FastAPI(title="Inquix Knowledge Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    fallback: bool
    latency_s: float


def log_query(record: dict) -> None:
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = state["assistant"].ask(req.question)
    except Exception as exc:
        # If the LLM call fails, degrade gracefully instead of crashing
        logger.exception("Assistant error")
        log_query({"question": req.question, "error": str(exc)})
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable.")

    log_query({"question": req.question, **result})
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        fallback=result["fallback"],
        latency_s=result["latency_s"],
    )