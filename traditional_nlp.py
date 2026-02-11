"""
Step 1: Traditional NLP predictions using standard Python libraries.

Tasks:
  1. sentiment  -> NLTK VADER
  2. ner        -> spaCy (en_core_web_sm)
  3. pos        -> NLTK
  4. topic      -> sklearn LDA
  5. language   -> langdetect

Run all:           python traditional_nlp.py --all
Run selected:      python traditional_nlp.py --tasks sentiment ner
List tasks:       python traditional_nlp.py --list
Output: results/*.csv
"""

import argparse
import os
import csv
import time
from tqdm import tqdm

task_times = {}  # task name -> seconds

AVAILABLE_TASKS = ["sentiment", "ner", "pos", "topic", "language"]


def _parse_args():
    p = argparse.ArgumentParser(description="Run selected traditional NLP tasks.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Run all tasks")
    g.add_argument("--tasks", nargs="+", choices=AVAILABLE_TASKS, metavar="TASK", help="Run only these tasks")
    g.add_argument("--list", action="store_true", help="List available tasks and exit")
    args = p.parse_args()
    if args.list:
        print("Available tasks:", ", ".join(AVAILABLE_TASKS))
        raise SystemExit(0)
    return AVAILABLE_TASKS if args.all else args.tasks


tasks = _parse_args()
os.makedirs("results", exist_ok=True)

# --- Imports and setup only for selected tasks ---
if "sentiment" in tasks or "pos" in tasks:
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt_tab", quiet=True)

if "ner" in tasks:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm' (one-time)...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
else:
    nlp = None

from custom_datasets import (
    sentiment_data, ner_data, pos_data,
    topic_data, language_data, language_ground_truth,
)


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> Saved {len(rows)} rows to {path}\n")


# ============================================================
# TASK 1: Sentiment Analysis (VADER)
# ============================================================
if "sentiment" in tasks:
    t0 = time.perf_counter()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    print("=" * 60)
    print("TASK 1: Sentiment Analysis (VADER)")
    print("=" * 60)
    sid = SentimentIntensityAnalyzer()
    sentiment_results = []
    for i, text in tqdm(enumerate(sentiment_data), total=len(sentiment_data), desc="Sentiment"):
        score = sid.polarity_scores(text)["compound"]
        label = "positive" if score >= 0 else "negative"
        sentiment_results.append({"id": i + 1, "text": text, "label": label, "score": round(score, 4)})
        print(f"  {i+1:>2}. [{label:>8}] {text[:65]}")
    save_csv("results/sentiment_traditional.csv", ["id", "text", "label", "score"], sentiment_results)
    task_times["sentiment"] = time.perf_counter() - t0
    print(f"  Time: {task_times['sentiment']:.2f}s\n")

# ============================================================
# TASK 2: Named Entity Recognition (spaCy)
# ============================================================
# Map spaCy labels to CoNLL-2003 types (PER, ORG, LOC, MISC) for alignment with gold
SPACY_TO_CONLL_TYPE = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",      # facility -> location
    "EVENT": "MISC",
    "PRODUCT": "MISC",
    "WORK_OF_ART": "MISC",
    "NORP": "MISC",
    "LANGUAGE": "MISC",
    "LAW": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "PERCENT": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
}

if "ner" in tasks:
    t0 = time.perf_counter()
    print("=" * 60)
    print("TASK 2: Named Entity Recognition (spaCy)")
    print("=" * 60)
    ner_results = []
    for i, text in tqdm(enumerate(ner_data), total=len(ner_data), desc="NER"):
        doc = nlp(text)
        # Output CoNLL-2003 types (PER, ORG, LOC, MISC) for fair comparison with gold
        parts = []
        for ent in doc.ents:
            gold_type = SPACY_TO_CONLL_TYPE.get(ent.label_, "MISC")
            parts.append(f"{ent.text} ({gold_type})")
        entities = "; ".join(parts) if parts else "NONE"
        ner_results.append({"id": i + 1, "text": text, "entities": entities})
        print(f"  {i+1:>2}. {text[:45]}...")
        print(f"      -> {entities}")
    save_csv("results/ner_traditional.csv", ["id", "text", "entities"], ner_results)
    task_times["ner"] = time.perf_counter() - t0
    print(f"  Time: {task_times['ner']:.2f}s\n")

# ============================================================
# TASK 3: POS Tagging (NLTK)
# ============================================================
if "pos" in tasks:
    t0 = time.perf_counter()
    import nltk
    print("=" * 60)
    print("TASK 3: POS Tagging (NLTK)")
    print("=" * 60)
    pos_results = []
    for i, text in tqdm(enumerate(pos_data), total=len(pos_data), desc="POS"):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        tags_str = " ".join(f"{w}/{t}" for w, t in tagged)
        pos_results.append({"id": i + 1, "text": text, "pos_tags": tags_str})
        print(f"  {i+1:>2}. {text[:45]}")
        print(f"      -> {tags_str[:80]}...")
    save_csv("results/pos_traditional.csv", ["id", "text", "pos_tags"], pos_results)
    task_times["pos"] = time.perf_counter() - t0
    print(f"  Time: {task_times['pos']:.2f}s\n")

# ============================================================
# TASK 4: Topic Modelling (sklearn LDA)
# ============================================================
if "topic" in tasks:
    t0 = time.perf_counter()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    print("=" * 60)
    print("TASK 4: Topic Modelling (sklearn LDA)")
    print("=" * 60)
    N_TOPICS = 4
    vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words="english")
    dtm = vectorizer.fit_transform(topic_data)
    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, max_iter=20)
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for idx, topic_vec in enumerate(lda.components_):
        top5 = [feature_names[j] for j in topic_vec.argsort()[:-6:-1]]
        topic_keywords[idx] = ", ".join(top5)
        print(f"  Topic {idx}: {topic_keywords[idx]}")
    print()
    doc_topics = lda.transform(dtm)
    topic_results = []
    for i, text in tqdm(enumerate(topic_data), total=len(topic_data), desc="Topic"):
        t = int(doc_topics[i].argmax())
        conf = round(float(doc_topics[i].max()), 4)
        topic_results.append({
            "id": i + 1, "text": text,
            "topic_id": t, "keywords": topic_keywords[t], "confidence": conf,
        })
        print(f"  {i+1:>2}. Topic {t} ({conf:.2f}): {text[:55]}...")
    save_csv(
        "results/topic_traditional.csv",
        ["id", "text", "topic_id", "keywords", "confidence"],
        topic_results,
    )
    task_times["topic"] = time.perf_counter() - t0
    print(f"  Time: {task_times['topic']:.2f}s\n")

# ============================================================
# TASK 5: Language Identification (langdetect)
# ============================================================
if "language" in tasks:
    t0 = time.perf_counter()
    from langdetect import detect
    print("=" * 60)
    print("TASK 5: Language Identification (langdetect)")
    print("=" * 60)
    LANG_MAP = {
        "en": "English", "fr": "French", "es": "Spanish",
        "de": "German", "it": "Italian", "pt": "Portuguese",
        "nl": "Dutch", "ru": "Russian", "ca": "Catalan",
        "ro": "Romanian", "af": "Afrikaans",
    }
    lang_results = []
    for i, text in tqdm(enumerate(language_data), total=len(language_data), desc="Language"):
        code = detect(text)
        predicted = LANG_MAP.get(code, code)
        actual = language_ground_truth[i]
        correct = predicted == actual
        lang_results.append({
            "id": i + 1, "text": text,
            "actual": actual, "predicted": predicted, "correct": correct,
        })
        mark = "Y" if correct else "X"
        print(f"  {i+1:>2}. [{mark}] Predicted={predicted:<10} Actual={actual:<10} | {text[:40]}")
    n_correct = sum(1 for r in lang_results if r["correct"])
    print(f"\n  Accuracy: {n_correct}/{len(lang_results)} ({100*n_correct/len(lang_results):.1f}%)")
    save_csv(
        "results/language_traditional.csv",
        ["id", "text", "actual", "predicted", "correct"],
        lang_results,
    )
    task_times["language"] = time.perf_counter() - t0
    print(f"  Time: {task_times['language']:.2f}s\n")

print("=" * 60)
print("DONE. Ran:", ", ".join(tasks))
if task_times:
    total = sum(task_times.values())
    for name, secs in task_times.items():
        print(f"  {name}: {secs:.2f}s")
    print(f"  Total: {total:.2f}s")
print("=" * 60)
