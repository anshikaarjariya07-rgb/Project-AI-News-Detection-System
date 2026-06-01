import random
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from include.text_processing import clean_text

FAKE_TEMPLATES = [
    "Scientists SHOCKED as {topic} defies all known laws — government hiding the truth!",
    "BREAKING: {topic} proven to cause {effect} — mainstream media SILENT",
    "Leaked documents reveal {topic} is a secret plot by {actor} — share before deleted!",
    "You won't BELIEVE what {topic} really does — doctors hate this!",
]
REAL_TEMPLATES = [
    "Researchers at {univ} publish findings on {topic} in peer-reviewed journal",
    "Government releases annual report on {topic}; experts cautiously optimistic",
    "Study involving {n} participants examines the relationship between {topic} and {effect}",
]
TOPICS  = ["vaccines", "climate change", "5G towers", "the economy", "renewable energy"]
EFFECTS = ["autism", "cancer", "mind control", "population decline"]
ACTORS  = ["the government", "Big Pharma", "the UN", "tech billionaires"]
UNIVS   = ["MIT", "Oxford", "Stanford", "Johns Hopkins"]
NS      = ["1,200", "4,500", "850"]

def _fill(template):
    return template.format(topic=random.choice(TOPICS), effect=random.choice(EFFECTS),
                           actor=random.choice(ACTORS), univ=random.choice(UNIVS), n=random.choice(NS))

def generate_synthetic_dataset(n_per_class: int = 800) -> pd.DataFrame:
    rows = []
    for _ in range(n_per_class): rows.append({"text": _fill(random.choice(FAKE_TEMPLATES)), "label": "FAKE"})
    for _ in range(n_per_class): rows.append({"text": _fill(random.choice(REAL_TEMPLATES)), "label": "REAL"})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

def build_new_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
    ])