import time
from groq import Groq
from config import settings

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def chat(
    model_id: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> dict:
    """
    Returns dict with keys: text, tokens_used, latency_ms, error
    """
    client = get_client()
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else None
        return {"text": text, "tokens_used": tokens, "latency_ms": round(latency_ms, 1), "error": None}
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"text": "", "tokens_used": None, "latency_ms": round(latency_ms, 1), "error": str(e)}
