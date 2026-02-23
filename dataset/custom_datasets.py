"""
Shared datasets for all NLP tasks.

All five tasks use standard benchmark datasets loaded via the `datasets`
library (Hugging Face):

  1. Sentiment analysis  -> SST-2      (20 000 sentences)
  2. NER                 -> CoNLL-2003 (1 000 sentences)
  3. POS tagging         -> CoNLL-2003 (same 1 000 sentences as NER)
  4. Topic classification -> AG News   (1 000 articles, 4 classes)
  5. Language ID          -> papluca/language-identification (1 000 texts, 20 languages)
"""

import random
from datasets import load_dataset

# Global random seed / RNG used for any shuffling in this module
DATASET_SEED = 42
DATASET_RNG = random.Random(DATASET_SEED)


def _build_sentiment_from_sst2(n_per_class: int = 10000) -> tuple[list[str], list[str]]:
    """
    Build a small sentiment dataset from the SST‑2 benchmark.

    We take the first `n_per_class` positive and negative sentences
    from the train split and then shuffle them with a fixed seed.
    """

    ds = load_dataset("stanfordnlp/sst2", split="train")

    pos_sents = [ex["sentence"] for ex in ds if ex["label"] == 1][:n_per_class]
    neg_sents = [ex["sentence"] for ex in ds if ex["label"] == 0][:n_per_class]

    texts: list[str] = []
    labels: list[str] = []

    for s in pos_sents:
        texts.append(s)
        labels.append("positive")

    for s in neg_sents:
        texts.append(s)
        labels.append("negative")

    # Mix positives and negatives but keep the pairing between
    # text and label consistent and deterministic across runs.
    indices = list(range(len(texts)))
    DATASET_RNG.shuffle(indices)

    texts_shuffled = [texts[i] for i in indices]
    labels_shuffled = [labels[i] for i in indices]

    return texts_shuffled, labels_shuffled


try:
    sentiment_data, sentiment_ground_truth = _build_sentiment_from_sst2()
except Exception as e:
    # If something goes wrong (no internet, etc.), fail loudly so you
    # notice that the standard benchmark dataset was not actually used.
    print("ERROR: failed to build sentiment_data from SST‑2 (stanfordnlp/sst2):", e)
    raise

# ============================================================
# 2. NER Dataset (CoNLL-2003, 1000 samples)
# ============================================================
# Load from raw CoNLL-2003 data (avoids Hugging Face dataset scripts deprecated in datasets>=4.0)
import io
import zipfile
import urllib.request

CONLL2003_ZIP_URL = "https://data.deepai.org/conll2003.zip"
CONLL2003_TRAIN_FILE = "train.txt"


def _ner_tag_str_to_type(tag: str) -> str | None:
    """Convert IOB2 string tag to type: 'B-PER' -> 'PER', 'O' -> None."""
    if not tag or tag == "O":
        return None
    if tag.startswith("B-") or tag.startswith("I-"):
        return tag[2:]
    return None


def _tokens_ner_str_to_entity_string(tokens: list, ner_tag_strs: list) -> str:
    """Convert tokens + list of string NER tags (e.g. 'B-PER', 'I-PER', 'O') to 'Entity (TYPE); ...'."""
    parts = []
    i = 0
    while i < len(tokens):
        tag = ner_tag_strs[i] if i < len(ner_tag_strs) else "O"
        etype = _ner_tag_str_to_type(tag)
        if etype is None:
            i += 1
            continue
        span_tokens = [tokens[i]]
        i += 1
        while i < len(tokens) and i < len(ner_tag_strs) and _ner_tag_str_to_type(ner_tag_strs[i]) == etype:
            span_tokens.append(tokens[i])
            i += 1
        text = " ".join(span_tokens)
        parts.append(f"{text} ({etype})")
    return "; ".join(parts) if parts else "NONE"


def _parse_conll2003_file(lines: list[str]) -> list[tuple[list[str], list[str], list[str]]]:
    """Parse CoNLL-2003 format: each line 'word pos chunk ner', blank line = sentence end.
    Returns list of (tokens, pos_tag_strs, ner_tag_strs) per sentence.
    """
    sentences = []
    tokens = []
    pos_tags = []
    ner_tags = []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("-DOCSTART-") or line == "":
            if tokens:
                sentences.append((tokens, pos_tags, ner_tags))
                tokens = []
                pos_tags = []
                ner_tags = []
            continue
        parts = line.split()
        if len(parts) >= 4:
            tokens.append(parts[0])
            pos_tags.append(parts[1])
            ner_tags.append(parts[3])
    if tokens:
        sentences.append((tokens, pos_tags, ner_tags))
    return sentences


def _build_conll2003_ner_and_pos(n_samples: int = 1000) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Build NER and POS datasets from CoNLL-2003 train split (same 1000 sentences).
    Returns (texts, ner_gold_entities, pos_data_texts, pos_gold_tags).
    pos_gold_tags[i] = "w1/T1 w2/T2 ..." (Penn Treebank format).
    """
    with urllib.request.urlopen(CONLL2003_ZIP_URL, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data), "r") as z:
        names = z.namelist()
        train_name = next((n for n in names if n.endswith("train.txt") or n == "train.txt"), None)
        if not train_name:
            raise FileNotFoundError("train.txt not found in conll2003.zip")
        with z.open(train_name) as f:
            lines = [ln.decode("utf-8", errors="replace") for ln in f]
    sentences = _parse_conll2003_file(lines)
    n = min(n_samples, len(sentences))
    indices = list(range(len(sentences)))
    DATASET_RNG.shuffle(indices)
    indices = indices[:n]

    texts = []
    gold_entities = []
    pos_gold_tags = []
    for idx in indices:
        toks, pos_strs, ner_strs = sentences[idx]
        text = " ".join(toks)
        entity_str = _tokens_ner_str_to_entity_string(toks, ner_strs)
        pos_str = " ".join(f"{w}/{t}" for w, t in zip(toks, pos_strs))
        texts.append(text)
        gold_entities.append(entity_str)
        pos_gold_tags.append(pos_str)
    return texts, gold_entities, texts, pos_gold_tags


try:
    ner_data, ner_ground_truth, _pos_texts, pos_ground_truth = _build_conll2003_ner_and_pos(1000)
    pos_data = _pos_texts  # same 1000 sentences as NER
except Exception as e:
    print("ERROR: failed to build ner_data/pos_data from CoNLL-2003:", e)
    raise

# ============================================================
# 3. POS Tagging Dataset (1000 samples from CoNLL-2003, same as NER)
# ============================================================
# pos_data and pos_ground_truth are set above

# ============================================================
# 4. Topic Classification Dataset (AG News, 1000 samples)
# ============================================================

AG_NEWS_LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def _build_topic_from_ag_news(n_samples: int = 1000) -> tuple[list[str], list[str]]:
    """
    Build a topic classification dataset from the AG News benchmark (test split).

    AG News has 4 classes: World, Sports, Business, Sci/Tech.
    We take `n_samples` articles, shuffle deterministically, and return
    (texts, gold_labels).
    """
    ds = load_dataset("fancyzhx/ag_news", split="test")

    texts: list[str] = []
    labels: list[str] = []
    for ex in ds:
        texts.append(ex["text"])
        labels.append(AG_NEWS_LABEL_MAP[ex["label"]])

    # Deterministic shuffle then take first n_samples
    indices = list(range(len(texts)))
    DATASET_RNG.shuffle(indices)
    indices = indices[:n_samples]

    return [texts[i] for i in indices], [labels[i] for i in indices]


try:
    topic_data, topic_ground_truth = _build_topic_from_ag_news(1000)
except Exception as e:
    print("ERROR: failed to build topic_data from AG News (fancyzhx/ag_news):", e)
    raise

# ============================================================
# 5. Language Identification Dataset (papluca/language-identification, 1000 samples)
# ============================================================

LANGID_CODE_TO_NAME = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


def _build_language_identification(n_samples: int = 1000) -> tuple[list[str], list[str]]:
    """
    Build a language identification dataset from the papluca/language-identification
    benchmark (test split, 10 000 samples across 20 languages).

    Returns (texts, gold_language_names).
    """
    ds = load_dataset("papluca/language-identification", split="test")

    texts: list[str] = []
    labels: list[str] = []
    for ex in ds:
        texts.append(ex["text"])
        labels.append(LANGID_CODE_TO_NAME.get(ex["labels"], ex["labels"]))

    indices = list(range(len(texts)))
    DATASET_RNG.shuffle(indices)
    indices = indices[:n_samples]

    return [texts[i] for i in indices], [labels[i] for i in indices]


try:
    language_data, language_ground_truth = _build_language_identification(1000)
except Exception as e:
    print("ERROR: failed to build language_data from papluca/language-identification:", e)
    raise
