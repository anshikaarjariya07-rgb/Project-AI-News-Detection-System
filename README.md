# ⬡ AI-Powered Fake News Detection System

A lightweight, desktop-based Python application that uses Machine Learning to analyze text patterns and classify articles as real or fake news. It features a responsive graphical interface, background processing threads, and automated database logging.

---

## 📝 Project Description

This system addresses the challenge of online misinformation by analyzing linguistic patterns within textual data. Instead of cross-referencing news articles against a static fact-checking database, the system uses natural language processing (NLP) and statistical classification to evaluate the structural vocabulary of an article and calculate truth probabilities in real time.

### Key Features:
*   **Predictive Engine:** Combines a **TF-IDF Vectorizer** with a balanced **Logistic Regression** model to identify patterns typical of misinformation.
*   **Smooth Performance:** Offloads heavy data generation and model training to asynchronous background threads so the user interface never freezes.
*   **Text Preprocessing:** Automatically cleans input text by standardizing casing, removing URLs, and stripping non-alphabetic noise characters.
*   **Resilient Storage:** Connects to a local MongoDB instance to log calculation history. If the database is missing, it smoothly switches to an **Offline Mode** to function completely in local system memory without crashing.

---

## 📂 Project Repository Structure

```text
fake-news-detector/
│
├── include/
│   ├── __init__.py           # Package marker file
│   ├── database.py           # MongoDB transaction layer & fallback logic
│   └── text_processing.py    # Text standardization & regex cleanup engines
│
├── src/
│   ├── app.py                # Tkinter graphical layout & control handlers
│   └── ml_pipeline.py        # Synthetic dataset generators & model training
│
├── docs/
│   ├── Algorithm.txt         # Logical mathematical pipelines
│   └── Flowchart.txt         # Plain-text mapping of system operations
│
├── main.py                   # Master entry point initiation script
├── README.md                 # Primary project landing page document
└── requirements.txt          # Third-party dependency library manifest
