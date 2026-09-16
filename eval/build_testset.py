"""Build the RAGAS test set from ElternLeben.de articles.

For each article in ARTICLES, an LLM writes one realistic parent question and a
reference answer grounded strictly in that article. The result is written to
eval/testset.json and should be reviewed by hand before being committed.

Usage:
    python eval/build_testset.py
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from rag import DATA_DIR, clean_text, extract_url  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI()

OUTPUT_FILE = os.path.join(ROOT, "eval", "testset.json")
MODEL = "gpt-4o"

# Spread across age groups and topics so scores are not dominated by baby sleep
ARTICLES = [
    "baby/babyschlaf/baby-pucken.md",
    "baby/babyschlaf/wie-viel-schlaf-braucht-ein-baby.md",
    "baby/babynahrung/beikost-einfuehren.md",
    "baby/gesundes-baby/zahnen.md",
    "baby/gesundes-baby/ploetzlicher-kindstod.md",
    "baby/entwicklung-baby/drei-monats-koliken.md",
    "baby/entwicklung-baby/wenn-babys-fremdeln.md",
    "baby/entwicklung-baby/ab-wann-duerfen-babys-medien-nutzen.md",
    "kleinkind/kita/eingewoehnung-in-kindergarten.md",
    "kleinkind/entwicklung-foerderung/was-ist-die-trotzphase-oder-autonomiephase.md",
    "kleinkind/trocken-werden/toepfchen-oder-toilettenaufsatz-was-ist-besser.md",
    "schwangerschaft/gesunde-schwangerschaft/tipps-gegen-uebelkeit.md",
    "schwangerschaft/geburt/wochenbettdepression.md",
    "schulkind/herausforderungen-in-der-schule/mobbing.md",
    "schulkind/lernen/hausaufgaben.md",
    "teenager/pubertaet/kommunikation-in-der-pubertaet.md",
    "gesundheit-ernaehrung/mentale-gesundheit/burnout-bei-eltern.md",
    "gesundheit-ernaehrung/gesunde-ernaehrung-fuer-kinder/gesunde-getraenke.md",
    "erziehung-und-foerderung/sprachentwicklung-von-kindern/stottern.md",
    "haeufige-fragen/trotz-wut/mein-kind-hat-starke-wutanfaelle-wie-soll-ich-reagieren.md",
]

PROMPT = """Du erstellst Testdaten für einen Eltern-Chatbot.

Lies den folgenden Artikel von ElternLeben.de und schreibe:
1. "question": Eine realistische Frage, wie ein Elternteil sie in einen Chat tippen würde
   (natürliche Sprache, konkrete Situation, NICHT den Artikeltitel umformulieren).
   Die Frage muss mit dem Artikel beantwortbar sein.
2. "reference": Eine Referenzantwort in 3–5 Sätzen, die NUR Fakten und Empfehlungen
   aus dem Artikel enthält.

Antworte als JSON-Objekt mit den Schlüsseln "question" und "reference".

Artikel:
{article}"""


def build_sample(relative_path: str) -> dict:
    with open(os.path.join(DATA_DIR, relative_path), encoding="utf-8") as f:
        raw = f.read()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(article=clean_text(raw))}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content)
    return {
        "question": data["question"],
        "reference": data["reference"],
        "source_url": extract_url(raw),
        "source_file": relative_path,
    }


def main():
    samples = []
    for i, path in enumerate(ARTICLES, 1):
        print(f"[{i}/{len(ARTICLES)}] {path}")
        samples.append(build_sample(path))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(samples)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
