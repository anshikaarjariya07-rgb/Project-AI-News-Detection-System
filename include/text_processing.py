import re

def clean_text(text: str) -> str:
    """Strips URLs, numbers, punctuation, and downcases strings cleanly."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text