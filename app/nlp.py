import os, torch
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM)

CLF_REPO = os.getenv("CLF_REPO")   # HasibSharif/parsbert-dari-classifier
SUM_REPO = os.getenv("SUM_REPO")   # HasibSharif/dari-summarizer-mt5-small
HF_TOKEN = os.getenv("HF_TOKEN")

tok_clf = AutoTokenizer.from_pretrained(CLF_REPO, token=HF_TOKEN)
clf     = AutoModelForSequenceClassification.from_pretrained(CLF_REPO, token=HF_TOKEN).eval()

tok_sum = AutoTokenizer.from_pretrained(SUM_REPO, token=HF_TOKEN)
summ    = AutoModelForSeq2SeqLM.from_pretrained(SUM_REPO, token=HF_TOKEN).eval()

LABEL_FA = {                             # add whichever you use
    "Agriculture": "زراعت",
    "Art-Culture": "هنر و فرهنگ",
    "Banking-Insurance": "اقتصاد",
    "Economy": "اقتصاد",
    "Education-University": "تحصیلات",
    "Health": "صحت",
    "Industry": "صنعت",
    "International": "بین‌الملل",
    "Local": "داخلی",
    "Oil-Energy": "نفت و انرژی",
    "Politics": "سیاست",
    "Research": "تحقیق",
    "Roads-Urban": "شهرسازی",
    "Science-Technology": "فناوری",
    "Society": "جامعه",
    "Sports": "ورزش",
    "Tourism": "گردشگری",
    "Transportation": "ترانسپورت",
}

def classify(text: str) -> str:
    inp = tok_clf(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        pred = clf(**inp).logits.softmax(-1).argmax(-1).item()
    id2label = clf.config.id2label
    eng = id2label[str(pred)]
    return LABEL_FA.get(eng, eng)

def summarise(text: str) -> str:
    enc = tok_sum("summarize: "+text, return_tensors="pt",
                  truncation=True, max_length=512)
    with torch.no_grad():
        out = summ.generate(**enc, max_length=128, num_beams=4,
                            no_repeat_ngram_size=3, early_stopping=True)
    return tok_sum.decode(out[0], skip_special_tokens=True)
