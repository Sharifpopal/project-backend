# import os, torch
# from transformers import (AutoTokenizer,
#                           AutoModelForSequenceClassification,
#                           AutoModelForSeq2SeqLM)

# CLF_REPO = os.getenv("CLF_REPO")   # HasibSharif/parsbert-dari-classifier
# SUM_REPO = os.getenv("SUM_REPO")   # HasibSharif/dari-summarizer-mt5-small
# HF_TOKEN = os.getenv("HF_TOKEN")

"""
app/nlp.py – call Hugging Face Inference API instead of
loading the models into memory.
"""

import os, requests, time
from typing import Dict, List

# Environment variables provided via Render dashboard
HF_TOKEN  = os.environ["HF_TOKEN"]                # secret
CLF_REPO  = os.environ["CLF_REPO"]                # e.g. HasibSharif/news-classifier-pb
SUM_REPO  = os.environ["SUM_REPO"]                # e.g. HasibSharif/dari-summarizer-mt5-small

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
TIMEOUT = 20     # seconds


# ───────────────────────────────────────── classify ──
def _hf_post(model_id: str, payload: Dict) -> List[Dict]:
    """Robust POST with automatic retry while HF spins up container."""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    for _ in range(4):                          # up to 3 retries
        r = requests.post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 503:                # model still loading
            time.sleep(5)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"HF API failed for {model_id}: {r.text[:200]}")
# tok_clf = AutoTokenizer.from_pretrained(CLF_REPO, token=HF_TOKEN)
# clf     = AutoModelForSequenceClassification.from_pretrained(CLF_REPO, token=HF_TOKEN).eval()

# tok_sum = AutoTokenizer.from_pretrained(SUM_REPO, token=HF_TOKEN)
# summ    = AutoModelForSeq2SeqLM.from_pretrained(SUM_REPO, token=HF_TOKEN).eval()

# LABEL_FA = {                             # add whichever you use
#     "Agriculture": "زراعت",
#     "Art-Culture": "هنر و فرهنگ",
#     "Banking-Insurance": "اقتصاد",
#     "Economy": "اقتصاد",
#     "Education-University": "تحصیلات",
#     "Health": "صحت",
#     "Industry": "صنعت",
#     "International": "بین‌الملل",
#     "Local": "داخلی",
#     "Oil-Energy": "نفت و انرژی",
#     "Politics": "سیاست",
#     "Research": "تحقیق",
#     "Roads-Urban": "شهرسازی",
#     "Science-Technology": "فناوری",
#     "Society": "جامعه",
#     "Sports": "ورزش",
#     "Tourism": "گردشگری",
#     "Transportation": "ترانسپورت",
# }

def classify(text: str) -> str:
    """
    Returns the *top* predicted label (string) from the classifier repo.
    """
    resp = _hf_post(CLF_REPO, {
        "inputs": text,
        "options": {"wait_for_model": True}
    })
    # HF returns list of dicts [{'label': 'Economy', 'score': 0.998}, …]
    return resp[0]["label"]


# ──────────────────────────────────────── summarise ──
def summarise(text: str) -> str:
    prompt = "summarize: " + text
    resp = _hf_post(SUM_REPO, {
        "inputs": prompt,
        "parameters": {
            "max_length": 128,
            "num_beams": 4,
            "no_repeat_ngram_size": 4,
        },
        "options": {"wait_for_model": True}
    })
    return resp[0]["generated_text"]
# def classify(text: str) -> str:
#     inp = tok_clf(text, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         pred = clf(**inp).logits.softmax(-1).argmax(-1).item()
#     id2label = clf.config.id2label
#     eng = id2label[str(pred)]
#     return LABEL_FA.get(eng, eng)

# def summarise(text: str) -> str:
#     enc = tok_sum("summarize: "+text, return_tensors="pt",
#                   truncation=True, max_length=512)
#     with torch.no_grad():
#         out = summ.generate(**enc, max_length=128, num_beams=4,
#                             no_repeat_ngram_size=3, early_stopping=True)
#     return tok_sum.decode(out[0], skip_special_tokens=True)






