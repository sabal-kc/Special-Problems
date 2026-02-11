"""
Shared datasets for all NLP tasks.

For **sentiment analysis** we use the standard Stanford Sentiment Treebank 2
(SST‑2) dataset via the `datasets` library (10,000 positive + 10,000 negative
= 20,000 sentences by default). This is the benchmark commonly used in
sentiment comparison papers.

The other tasks still use small hand-written examples to keep the
pipeline light-weight.
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
# 4. Topic Modelling Dataset (20 samples)
# ============================================================
topic_data = [
    "The stock market reached an all-time high today with major gains in tech stocks.",
    "The new smartphone features a 108MP camera and 5G connectivity.",
    "The football team won the championship after a thrilling final match.",
    "Scientists published a breakthrough study on climate change effects.",
    "The central bank raised interest rates to combat inflation.",
    "A new artificial intelligence model can generate realistic images.",
    "The basketball player scored 50 points in last night's game.",
    "Researchers found a potential new treatment for Alzheimer's disease.",
    "The government announced new economic stimulus measures.",
    "The latest laptop comes with a powerful M3 chip and 24GB RAM.",
    "The tennis player won her fifth Grand Slam title.",
    "A new study reveals the impact of microplastics on ocean ecosystems.",
    "Cryptocurrency prices surged after new regulatory clarity.",
    "Virtual reality headsets are becoming more affordable for consumers.",
    "The soccer World Cup drew millions of viewers worldwide.",
    "Marine biologists discovered a new coral reef in the Pacific Ocean.",
    "The tech company reported record quarterly earnings.",
    "Wearable health devices can now monitor blood glucose levels.",
    "The Olympic games featured several new world records.",
    "Astronomers detected a potentially habitable exoplanet.",
]

# ============================================================
# 5. Language Identification Dataset (20 samples)
# ============================================================
language_data = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "Guten Tag, wie geht es Ihnen?",
    "Ciao, come stai?",
    "The weather is beautiful today.",
    "Je suis très content de vous voir.",
    "Me gusta mucho la comida española.",
    "Ich lerne Deutsch seit zwei Jahren.",
    "La pizza italiana è la migliore del mondo.",
    "This is a simple English sentence.",
    "Les enfants jouent dans le parc.",
    "El gato está durmiendo en el sofá.",
    "Das Buch ist sehr interessant.",
    "Mi piace viaggiare in Italia.",
    "Programming is fun and challenging.",
    "La vie est belle quand on est heureux.",
    "Necesito comprar leche y pan.",
    "Die Musik ist wunderschön.",
    "Roma è una città bellissima.",
]

language_ground_truth = [
    "English", "French", "Spanish", "German", "Italian",
    "English", "French", "Spanish", "German", "Italian",
    "English", "French", "Spanish", "German", "Italian",
    "English", "French", "Spanish", "German", "Italian",
]
