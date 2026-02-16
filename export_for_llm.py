"""
Step 2: Export datasets as CSV *without labels* so you can paste them
into Google AI Studio alongside the prompts from prompts.md.

Run:  python export_for_llm.py
Output: llm_input/*.csv   (one per task)
"""

import os
import csv

from custom_datasets import (
    sentiment_data, ner_data, pos_data,
    topic_data, topic_ground_truth,
    language_data, language_ground_truth,
)

os.makedirs("llm_input", exist_ok=True)


def write_simple_csv(path, rows):
    """Write a list of (id, text) rows to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text"])
        w.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


# 1. Sentiment Analysis
write_simple_csv(
    "llm_input/sentiment_input_min.csv",
    [(i + 1, t) for i, t in enumerate(sentiment_data)],
)

# # 2. NER
# write_simple_csv(
#     "llm_input/ner_input.csv",
#     [(i + 1, t) for i, t in enumerate(ner_data)],
# )

# # 3. POS Tagging
# write_simple_csv(
#     "llm_input/pos_input.csv",
#     [(i + 1, t) for i, t in enumerate(pos_data)],
# )

# 4. Topic Modelling
# write_simple_csv(
#     "llm_input/topic_input.csv",
#     [(i + 1, t) for i, t in enumerate(topic_data)],
# )

# # 5. Language Identification
# write_simple_csv(
#     "llm_input/language_input.csv",
#     [(i + 1, t) for i, t in enumerate(language_data)],
# )

print("\nAll input CSVs saved to llm_input/ folder.")
print("Open prompts.md for the prompts to use in Google AI Studio.")
