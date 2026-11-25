import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.retrieve import Retriever

load_dotenv()  # loads GEMINI_API_KEY from .env

MODEL = "gemini-2.5-flash"  # if you get "model not found", pick another from AI Studio
MIN_SCORE = 0.30  # below this, we don't trust the retrieval; tune it later
FALLBACK_ANSWER = (
    "I don't have enough information to answer that. "
    "Please contact the team directly."
)

SYSTEM_PROMPT = """You answer customer questions for a business.
Use ONLY the context provided. If the context does not contain the answer, say you don't know.
Keep answers short and clear. At the end, cite the source file names you used in square brackets."""


class Assistant:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(api_version="v1beta"),
        )

    def _generate(self, prompt: str):
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=300,
        )
        # The free tier has rate limits, so retry a few times if we hit one
        for attempt in range(3):
            try:
                return self.client.models.generate_content(
                    model=MODEL, contents=prompt, config=config
                )
            except Exception as exc:
                if attempt == 2 or "429" not in str(exc):
                    raise
                time.sleep(15)

    def ask(self, question: str, k: int = 3) -> dict:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=k)

        # Fallback: weak retrieval means we shouldn't guess
        if not hits or hits[0]["score"] < MIN_SCORE:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "fallback": True,
                "latency_s": round(time.perf_counter() - start, 3),
                "input_tokens": 0,
                "output_tokens": 0,
            }

        context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        response = self._generate(f"Context:\n{context}\n\nQuestion: {question}")
        usage = response.usage_metadata

        return {
            "answer": response.text or "",
            "sources": sorted({h["source"] for h in hits}),
            "fallback": False,
            "latency_s": round(time.perf_counter() - start, 3),
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
        }


if __name__ == "__main__":
    assistant = Assistant()
    for q in ["How long does a project take?", "When do I pay?", "Do you sell cars?"]:
        result = assistant.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"   sources={result['sources']} fallback={result['fallback']} "
              f"latency={result['latency_s']}s tokens={result['input_tokens']}/{result['output_tokens']}")