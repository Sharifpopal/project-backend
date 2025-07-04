import json, threading, pathlib, datetime
from typing import List, Dict

DATA_FILE = pathlib.Path("data/articles.json")
DATA_FILE.parent.mkdir(exist_ok=True, parents=True)

_lock = threading.Lock()
_articles: List[Dict] = []          # in-memory cache (latest first)


# ────────────────────────────────────────────────────────────
def _load_from_disk() -> None:
    """Load existing articles at start-up (if file exists)."""
    global _articles
    if DATA_FILE.exists():
        with DATA_FILE.open(encoding="utf-8") as fh:
            _articles = json.load(fh)
    else:
        _articles = []


def _save_to_disk() -> None:
    """Persist current cache to disk atomically."""
    tmp = DATA_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_articles, fh, ensure_ascii=False, indent=2)
    tmp.replace(DATA_FILE)


# ────────────────────────────────────────────────────────────
def add_articles(new_items: List[Dict]) -> int:
    """
    Insert new article dicts (dict must include at least 'url').
    Prevent duplicates by URL.  Returns number of articles added.
    """
    global _articles
    with _lock:
        existing_urls = {a["url"] for a in _articles}
        fresh = [a for a in new_items if a["url"] not in existing_urls]
        # newest on top
        for art in reversed(fresh):
            utc_now = datetime.datetime.utcnow().replace(microsecond=0)
            art["ingested_at"] = utc_now.isoformat() + "Z"
            _articles.insert(0, art)
        if fresh:
            _save_to_disk()
        return len(fresh)


def list_articles(limit: int | None = None) -> List[Dict]:
    with _lock:
        return _articles[:limit] if limit else list(_articles)


# initial load when module is imported
_load_from_disk()
