"""
top_topics.py  – scrape the #1 story from four Afghan-related sources,
                 extract full article data with newspaper3k,
                 render them into latest_news.html

deps:
    pip install requests beautifulsoup4 newspaper3k python-dateutil
"""

import os, datetime, requests, bs4
from newspaper import Article
from dateutil import tz
import feedparser
import json

# ────────────────────────────────────────────────────────────────
BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,"
              "application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fa,en-US;q=0.9,en;q=0.8",
    # gzip only – no “br”, so servers won’t send Brotli
    "Accept-Encoding": "gzip, deflate",
}
HEADERS = BASE_HEADERS

SELECTORS = {
    "Tolo News": (
        "https://tolonews.com/fa/",
        "h2.title-top-post-tolonews a",
        lambda a: "https://tolonews.com" + a["href"],
        BASE_HEADERS,                 # ← ordinary headers
    ),
    "Ariana News": (
        "https://www.ariananews.af/fa/",          # final slash matters
        "section#mvp-feat5-wrap a[rel='bookmark']",
        lambda a: a["href"],
        BASE_HEADERS,                 # gzip, no Brotli → works
    ),
    "RTA": (
        "https://rta.af/fa/home/",                # static template
        "a[rel='bookmark']",                      # first bookmark link
        lambda a: a["href"],
        {**BASE_HEADERS,
         # extra disguise in case Mod_Security is picky
         "Referer": "https://rta.af/",
        },
    ),
    "BBC Persian (افغانستان)": (
        "https://www.bbc.com/persian",
        "ul[data-testid='topic-promos'] li:first-child h3 a",
        lambda a: a["href"],
        BASE_HEADERS,
    ),
}



def get_html(url, headers, timeout=15):
    """
    Fetch URL with given headers.
    If 406 / Mod_Security, retry with minimal headers.
    Returns HTML text or raises.
    """
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code == 406:                 # “Not Acceptable” – RTA
        h2 = {k: v for k, v in headers.items() if k != "Accept-Encoding"}
        h2["Accept-Encoding"] = "identity"   # ask for raw, uncompressed
        r = requests.get(url, headers=h2, timeout=timeout)
    r.raise_for_status()
    return r.text

def top_link(source):
    url, css, href_fn, hdrs = SELECTORS[source]
    try:
        html  = get_html(url, hdrs)
        a_tag = bs4.BeautifulSoup(html, "html.parser").select_one(css)
        if not a_tag:
            raise ValueError("selector returned None")
        return href_fn(a_tag)
    except Exception:
        # ----- fallback for RTA: use their RSS feed -----
        if source == "RTA":
            feed = feedparser.parse("https://rta.af/fa/feed/")
            return feed.entries[0].link
        raise   

def fetch_article(url):
    art = Article(url, language="fa", browser_user_agent=HEADERS["User-Agent"])
    art.download();  art.parse()
    try:          art.nlp()     # yields summary/keywords when possible
    except:       pass
    return {
        "url":        url,
        "title":      art.title or "— بدون عنوان —",
        "text":       art.text.replace("\n", "<br>"),
        "summary":    getattr(art, "summary", "") or art.text[:280] + "…",
        "top_img":    art.top_image,
        "publish_dt": art.publish_date,
    }


def run(backend_url: str):
    """
    1. Scrape top article from each source
    2. POST each raw article dict to /ingest on the backend
    """
    collected = []
    for src in SELECTORS:
        try:
            link = top_link(src)
            art  = fetch_article(link)
            art["source"] = src           # add source name
            collected.append(art)
            print(f"✔ scraped {src}")
        except Exception as e:
            print(f"✖ {src}: {e}")

    if not collected:
        print("Nothing scraped; aborting.")
        return

    # POST in one batch (you could loop post-by-post instead)
    try:
        r = requests.post(
            f"{backend_url.rstrip('/')}/ingest",
            headers={"Content-Type": "application/json"},
            data=json.dumps(collected),
            timeout=30
        )
        r.raise_for_status()
        print(f"✅ Sent {len(collected)} articles → {backend_url}/ingest")
    except Exception as err:
        print("🚨 Failed to send to backend:", err)