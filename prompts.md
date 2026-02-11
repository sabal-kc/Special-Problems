# Gemini Prompts for Google AI Studio

> For each task below, paste the corresponding CSV data from `llm_input/` into
> Google AI Studio along with the prompt. Copy the **entire output** (including
> headers) and save it as the specified file inside the `llm_output/` folder.

---

## Task 1: Sentiment Analysis

**Input file:** `llm_input/sentiment_input.csv`
**Save output as:** `llm_output/sentiment_llm.csv`

### Prompt

```
You are a sentiment analysis expert. I will give you a CSV with columns "id" and "text".
For each row, classify the sentiment as exactly one of: positive, negative.

Return your answer as a CSV with exactly two columns: id, label
Do NOT include any explanation or extra text. Only output the CSV (with header row).

Here is the data:

<PASTE THE CONTENTS OF sentiment_input.csv HERE>
```

---

## Task 2: Named Entity Recognition (NER)

**Input file:** `llm_input/ner_input.csv`
**Save output as:** `llm_output/ner_llm.csv`

### Prompt

```
You are a named entity recognition expert. I will give you a CSV with columns "id" and "text".
For each row, extract all named entities and their types.

Use exactly these four entity types (same as CoNLL-2003):
- PER: person names
- ORG: organizations (companies, agencies, etc.)
- LOC: locations (cities, countries, regions, buildings, facilities, natural locations)
- MISC: miscellaneous named entities that do not fit PER, ORG, or LOC (e.g. product names, events, nationalities, titles of works)

Return your answer as a CSV with exactly two columns: id, entities
Format each entity as "entity_text (TYPE)" and separate multiple entities with "; "
Example: "Joe Biden (PER); Washington (LOC); White House (LOC)"
If no entities are found, write "NONE".
Do NOT include any explanation or extra text. Only output the CSV (with header row).

Here is the data:

<PASTE THE CONTENTS OF ner_input.csv HERE>
```

---

## Task 3: POS Tagging

**Input file:** `llm_input/pos_input.csv`
**Save output as:** `llm_output/pos_llm.csv`

### Prompt

```
You are a POS tagging expert. I will give you a CSV with columns "id" and "text".
For each row, perform Part-of-Speech tagging on every token using Penn Treebank tagset
(e.g., NN, VB, JJ, DT, IN, NNP, RB, etc.).

Return your answer as a CSV with exactly two columns: id, pos_tags
Format the pos_tags column as: word1/TAG1 word2/TAG2 word3/TAG3 ...
Do NOT include any explanation or extra text. Only output the CSV (with header row).

Here is the data:

<PASTE THE CONTENTS OF pos_input.csv HERE>
```

---

## Task 4: Topic Modelling

**Input file:** `llm_input/topic_input.csv`
**Save output as:** `llm_output/topic_llm.csv`

### Prompt

```
You are a topic classification expert. I will give you a CSV with columns "id" and "text".
Each text belongs to one of these 4 topics: World, Sports, Business, Sci/Tech.

For each row, assign exactly one topic label from the list above.

Return your answer as a CSV with exactly two columns: id, topic
Do NOT include any explanation or extra text. Only output the CSV (with header row).

Here is the data:

<PASTE THE CONTENTS OF topic_input.csv HERE>
```

---

## Task 5: Language Identification

**Input file:** `llm_input/language_input.csv`
**Save output as:** `llm_output/language_llm.csv`

### Prompt

```
You are a language identification expert. I will give you a CSV with columns "id" and "text".
For each row, identify the language of the text.

Use exactly these full language names: Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese.

Return your answer as a CSV with exactly two columns: id, language
Do NOT include any explanation or extra text. Only output the CSV (with header row).

Here is the data:

<PASTE THE CONTENTS OF language_input.csv HERE>
```

---

## After You're Done

1. Create a folder called `llm_output/` in the project directory
2. Save each output as the filename specified above
3. Open `compare_results.ipynb` and run all cells to see the comparison
