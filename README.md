# Steam Game Reviews — Sentiment Analysis Pipeline

A full end-to-end NLP project that collects, processes, labels, and classifies sentiment in user-generated reviews from Steam, the world's largest PC gaming platform. The pipeline covers every stage from raw data collection through to benchmarking against a state-of-the-art transformer model.

---

## Project Overview

User reviews on Steam are informal, noisy, and highly domain-specific — full of gaming slang, memes, irony, and mixed-language text. This project builds a complete sentiment analysis system on top of that data, classifying each review as **positive**, **neutral**, or **negative**, and systematically evaluating how different design choices at each stage of the pipeline affect final model performance.

The dataset consists of approximately **4,000 reviews** scraped from the **20 most popular games** on Steam (200 reviews per game), spanning a range of genres including competitive shooters, MMOs, RPGs, and indie titles.

---

## Repository Structure

```text
.
├── Data scrapping/
│   ├── Steam_Scrapping.ipynb
│   └── STEAM_GAMES.CSV
├── Data preprocessing/
│   ├── task_2_social.py
│   ├── STEAM_GAMES_REDUCED.CSV
│   └── STEAM_GAMES_CLEAN.csv
├── Full code.ipynb
├── requirements.txt
└── README.md
```

---

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Some sections use external LLM APIs through environment variables. Do not hardcode API keys in notebooks or scripts — set them in your environment before running those cells.

---

## Pipeline Walkthrough

### 1. Data Collection
> Folder: `Data scrapping` — Output: `STEAM_GAMES.CSV`

Reviews were collected from the Steam API using response streaming, targeting the top 20 games by active player count. No pre-filtering was applied, so the raw corpus reflects the full diversity of real user-generated content — including single-word reviews, emoji-heavy text, non-English entries, garbled encodings, and community meme language. Each record retains a category tag identifying which game it came from, enabling per-genre performance analysis in later stages.

---

### 2. Text Preprocessing
> Folder: `Data preprocessing` — Output: `STEAM_GAMES_CLEAN.csv`

A modular preprocessing pipeline was designed with ten independent cleaning operations that can be toggled in any combination at runtime:

- **Translation** of non-English reviews into English
- **Encoding correction** to remove garbled and non-ASCII characters
- **Noise removal** to strip punctuation and special characters
- **Lowercasing** for vocabulary normalization
- **Number removal** to eliminate digits with no sentiment value
- **Spell correction** using a two-tier approach for common and unusual misspellings
- **Lemmatization** to reduce inflected word forms to their base form
- **Stopword removal**, with negation words (e.g., "not", "never", "wouldn't") deliberately preserved to protect sentiment-bearing phrases
- **Emoji removal** for cleaner token sequences
- **Post-cleaning validation** to drop reviews reduced to empty strings after processing

Three preprocessing schemes were defined and carried forward in parallel throughout the modeling phase:

| Scheme | Name | Description |
|---|---|---|
| A | Light | Minimal cleaning — preserves sentence structure and raw vocabulary |
| B | Standard | Moderate cleaning with lemmatization — reduces word variants while keeping content intact |
| C | Aggressive | Heavy cleaning — removes stopwords and corrects spelling to maximize signal density |

Example command to run the full preprocessing pipeline:

```bash
python "Data preprocessing/task_2_social.py" \
  --input "Data preprocessing/STEAM_GAMES_REDUCED.CSV" \
  --output "Data preprocessing/STEAM_GAMES_CLEAN.csv" \
  --translate --fix_encoding --remove_noise --lowercase \
  --remove_numbers --fix_spelling --lemmatize \
  --extract_tags --remove_stopwords --remove_emojis
```

---

### 3. Sentiment Modelling and Evaluation
> Notebook: `Full code.ipynb`

The main notebook covers the full modelling lifecycle:

**Ground Truth Labeling** — Since no pre-existing labels existed for this dataset, sentiment ground truth was generated using top-tier LLM APIs. Three independent prompts were applied to each review, producing three label votes per record. Inter-annotator agreement was measured using **Fleiss' Kappa**.

**Feature Representation** — Two techniques were used to convert preprocessed text into numerical vectors:
- **Bag-of-Words** — term frequency vectors over the full vocabulary; simple but high-dimensional and blind to word meaning.
- **GloVe Embeddings** — pre-trained dense vectors encoding semantic relationships; each review is represented as the average of its word vectors in a 100-dimensional space.

Both representations were computed for all three preprocessing schemes, yielding six distinct feature sets.

**Lexical Baselines** — Two rule-based models were built first:
- A **SentiWordNet classifier** that aggregates per-word sentiment scores from a linguistic resource.
- A **Bing Liu dictionary classifier** built from scratch using published positive and negative word lists, extended with negation handling so that phrases like "not good" are treated correctly.

**Machine Learning Models** — Trained across all six feature sets:
- **Naive Bayes**
- **Random Forest** (and Decision Tree variants)

After identifying the best configuration, **hyperparameter tuning** was applied and **dimensionality reduction** was used to address overfitting. A detailed **error analysis** was conducted on misclassified reviews.

**SOTA Benchmarking** — A transformer-based model from Hugging Face was evaluated on the identical test set and compared side-by-side with the optimized ML model using the full suite of metrics, including per-category breakdowns.

**Interactive App** — A Streamlit web application was built to allow real-time sentiment prediction from the trained model.

---

## Results

### Best Performing Model

The winning configuration was **Standard Preprocessing (Scheme B) + GloVe Embeddings + Random Forest**:

| Metric | Score |
|---|---|
| Accuracy | 75% |
| ROC-AUC | 0.8152 |

### Model Comparison

| Metric | Optimized Model | SOTA Transformer |
|---|---|---|
| Accuracy | 75% | Higher |
| Macro F1 | Moderate | Higher |
| ROC-AUC | 0.8152 | Higher |
| Neutral Class | Weakest class | Stronger |
| Inference Speed | Fast | Slow |
| Requires GPU | No | Recommended |

### Key Findings

- **Preprocessing depth must be calibrated.** Moderate cleaning outperformed both minimal and aggressive strategies. Over-processing degraded performance on short, informal reviews.
- **Semantic embeddings matter.** GloVe substantially and consistently outperformed Bag-of-Words across every configuration tested.
- **Neutral is the hardest class.** Both models struggled most with neutral reviews, which lack the strong lexical anchors that drive classification in either direction.
- **Gaming language is a domain-specific challenge.** Slang, meme references, and figurative expressions were the primary source of misclassification in both models.

---

## Notes

- Keep API keys outside the repository and never commit them.
- The main documented workflow uses only `Data scrapping`, `Data preprocessing`, and `Full code.ipynb`.
- Clear saved notebook outputs containing sensitive data before committing.