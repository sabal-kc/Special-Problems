## Task 1: Sentiment Analysis

- **Dataset**: SST-2 (`stanfordnlp/sst2`, Hugging Face).
- **Construction**:
  - Take the **train split**.
  - Collect the first `10 000` **positive** and `10 000` **negative** sentences.
  - Map labels to strings: `"positive"` and `"negative"`.
  - Shuffle deterministically with a fixed random seed, giving **20 000 sentences** in total.
- **Traditional method (`traditional_nlp.py`)**:
  - Uses **NLTK VADER** (`SentimentIntensityAnalyzer`) as the classifier.
  - For each text, compute the VADER `compound` score.
  - Predict `"positive"` if `compound >= 0`, otherwise `"negative"`.
  - Save results to `results/sentiment_traditional.csv` with columns:
    - `id` (1-based index)
    - `text`
    - `label` (predicted sentiment)
    - `score` (VADER compound score, rounded)

---

## Task 2: Named Entity Recognition (NER)

- **Dataset**: CoNLL-2003 train data (downloaded from `https://data.deepai.org/conll2003.zip`).
- **Construction**:
  - Parse the **train file** in CoNLL-2003 format (`word pos chunk ner` per line).
  - Randomly shuffle sentence indices with a fixed seed and pick **1 000 sentences**.
  - For each sentence, build:
    - `text`: the tokens joined with spaces.
    - `gold_entities`: string like `"Barack Obama (PER); London (LOC)"`, where tags are converted from IOB2 (`B-PER`, `I-ORG`, etc.) to entity types (`PER`, `ORG`, `LOC`, `MISC`); if no entities, `"NONE"`.
- **Traditional method (`traditional_nlp.py`)**:
  - Uses **spaCy** (`en_core_web_sm`) as the NER model.
  - For each `text`, run `nlp(text)` to get `doc.ents`.
  - Map spaCy labels to CoNLL types using `SPACY_TO_CONLL_TYPE`, e.g.:
    - `PERSON -> PER`, `ORG -> ORG`, `GPE/LOC/FAC -> LOC`, others -> `MISC`.
  - Build a string `"Entity (TYPE); ..."` similar to the gold format; if none, `"NONE"`.
  - Save results to `results/ner_traditional.csv` with columns:
    - `id`
    - `text`
    - `entities` (predicted entities as a single string)

---

## Task 3: POS Tagging

- **Dataset**: Same **1 000 sentences** sampled for NER from CoNLL-2003.
- **Construction**:
  - For each selected sentence:
    - `text`: tokens joined with spaces.
    - `pos_ground_truth`: string `"w1/T1 w2/T2 ..."` using the POS tags from CoNLL.
- **Traditional method (`traditional_nlp.py`)**:
  - Uses **NLTK** for tokenization and POS tagging.
  - For each `text`:
    - Tokenize with `nltk.word_tokenize`.
    - Tag with `nltk.pos_tag`.
    - Build `"word/TAG"` pairs and join with spaces into a single string.
  - Save results to `results/pos_traditional.csv` with columns:
    - `id`
    - `text`
    - `pos_tags` (predicted `"word/TAG"` sequence)

---

## Task 4: Topic Classification

- **Dataset**: **AG News** (`fancyzhx/ag_news`, Hugging Face).
- **Construction (evaluation set)**:
  - Use the **test split**.
  - Map integer labels to topic names via:
    - `0 -> World`, `1 -> Sports`, `2 -> Business`, `3 -> Sci/Tech`.
  - Deterministically shuffle with a fixed seed and take **1 000 articles**.
  - Outputs:
    - `topic_data`: article texts.
    - `topic_ground_truth`: topic names (`"World"`, `"Sports"`, `"Business"`, `"Sci/Tech"`).
- **Traditional method (`traditional_nlp.py`)**:
  - Trains a **TF‑IDF + Multinomial Naive Bayes** classifier on AG News.
  - Training:
    - Load **train split** of `fancyzhx/ag_news`.
    - Map labels with the same label map as above.
    - Vectorize texts with `TfidfVectorizer(max_features=50000, stop_words="english")`.
    - Train `MultinomialNB` on the TF‑IDF features.
  - Evaluation:
    - Transform `topic_data` with the fitted TF‑IDF.
    - Predict topic names for each of the 1 000 evaluation texts.
    - Compare predictions with `topic_ground_truth` to compute accuracy.
  - Save results to `results/topic_traditional.csv` with columns:
    - `id`
    - `text`
    - `actual` (gold topic label)
    - `predicted` (model topic label)
    - `correct` (`True`/`False`)

---

## Task 5: Language Identification

- **Dataset**: `papluca/language-identification` (Hugging Face).
- **Construction**:
  - Use the **test split** (10 000 samples across 20 languages).
  - For each example:
    - `text`: the sentence or short text.
    - Map the ISO language code (`ex["labels"]`, e.g. `\"en\"`, `\"fr\"`) to a full language name using `LANGID_CODE_TO_NAME`, e.g.:
      - `"en" -> "English"`, `"fr" -> "French"`, `"zh" -> "Chinese"`, etc.
  - Shuffle deterministically and take **1 000 texts**.
  - Outputs:
    - `language_data`: texts.
    - `language_ground_truth`: language names (20-way classification).
- **Traditional method (`traditional_nlp.py`)**:
  - Uses **`langdetect`** to predict a language code for each text.
  - Maps `langdetect` codes to the same 20 language names via `LANG_MAP`, including
    fallbacks for common mis‑detections (e.g. `"zh-cn"` and `"zh-tw"` -> `"Chinese"`).
  - Compares predicted language name to the gold `language_ground_truth`.
  - Save results to `results/language_traditional.csv` with columns:
    - `id`
    - `text`
    - `actual` (gold language name)
    - `predicted` (model language name)
    - `correct` (`True`/`False`)

