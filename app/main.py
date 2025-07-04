from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from app import models
from app.nlp import classify, summarise     #  <-- your inference helpers

app = FastAPI(
    title="News NLP Backend",
    version="0.1.0",
    description="Classify + summarise Afghan news articles."
)


# ───────────────────────────────
# Pydantic schemas
# ───────────────────────────────
class RawArticle(BaseModel):
    url:   HttpUrl
    title: str
    text:  str
    source: str
    top_img: Optional[HttpUrl] = None
    publish_dt: Optional[str]  = None    # ISO8601 str acceptable


class ProcessedArticle(RawArticle):
    category: str
    summary:  str


# ───────────────────────────────
# Routes
# ───────────────────────────────
@app.post("/ingest", status_code=202)
async def ingest(items: List[RawArticle]):
    """
    Receive raw scraped articles from GitHub Action / worker.
    For each article:
      • classify category
      • summarise
      • store in memory (and JSON file)
    """
    processed: List[dict] = []
    for it in items:
        try:
            cat  = classify(it.title + "\n" + it.text)
            summ = summarise(it.text)
            art_dict = it.dict()
            art_dict.update({"category": cat, "summary": summ})
            processed.append(art_dict)
        except Exception as e:
            # log error but continue
            print("NLP failed:", e, "URL:", it.url)

    added = models.add_articles(processed)
    return {"received": len(items), "stored": added}


@app.get("/news", response_model=List[ProcessedArticle])
async def get_news(limit: int | None = 50):
    """
    Front-end fetches latest processed articles.
    Default returns 50 newest (sorted by ingest time).
    """
    return models.list_articles(limit=limit)


@app.get("/")
async def root():
    return {"status": "ok", "message": "News NLP backend is live."}
