import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

# Download required NLTK resources
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# ─────────────────────────────────────────────
# 1. POS TAG CONVERSION
# ─────────────────────────────────────────────
def get_wordnet_pos(treebank_tag):
    """Convert Penn Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None  # Skip words we can't map

# ─────────────────────────────────────────────
# 2. SENTIWORDNET SCORE LOOKUP
# ─────────────────────────────────────────────
def get_sentiment_score(word, pos):
    """
    Look up a word's sentiment scores in SentiWordNet.
    Returns (positive_score, negative_score, objective_score).
    Uses the first (most common) synset for the given POS.
    """
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos=pos)

    synsets = list(swn.senti_synsets(lemma, pos))
    if not synsets:
        return 0.0, 0.0, 1.0   # neutral / objective

    # Average over all synsets for this word+POS (or just take first)
    pos_score = np.mean([s.pos_score() for s in synsets])
    neg_score = np.mean([s.neg_score() for s in synsets])
    obj_score = np.mean([s.obj_score() for s in synsets])
    return pos_score, neg_score, obj_score

# ─────────────────────────────────────────────
# 3. NEGATION HANDLING
# ─────────────────────────────────────────────
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nobody",
    "nothing", "nowhere", "hardly", "scarcely", "barely",
    "n't", "nt", "without", "cannot", "can't", "won't",
    "isn't", "aren't", "wasn't", "weren't", "doesn't",
    "don't", "didn't", "hasn't", "haven't", "hadn't"
}

def apply_negation(tokens):
    """
    Flip sentiment for words appearing within a window after a negation token.
    Returns a list of (token, negated: bool) tuples.
    """
    WINDOW = 3          # how many words after negation are affected
    negated_flags = [False] * len(tokens)
    neg_countdown = 0

    for i, token in enumerate(tokens):
        if token.lower() in NEGATION_WORDS:
            neg_countdown = WINDOW
        elif neg_countdown > 0:
            negated_flags[i] = True
            neg_countdown -= 1

    return list(zip(tokens, negated_flags))

# ─────────────────────────────────────────────
# 4. MAIN CLASSIFIER
# ─────────────────────────────────────────────
def sentiwordnet_classify(text, threshold=0.05):
    """
    Classify a text as Positive, Negative, or Neutral using SentiWordNet.

    Parameters
    ----------
    text       : raw input string
    threshold  : minimum net score to assign pos/neg label (avoids noise)

    Returns
    -------
    dict with keys: label, net_score, pos_score, neg_score
    """
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)                       # [(word, POS), ...]
    token_neg_pairs = apply_negation(tokens)       # [(word, is_negated), ...]

    total_pos, total_neg = 0.0, 0.0
    scored_words = 0

    for (word, treebank_pos), (_, is_negated) in zip(tagged, token_neg_pairs):
        wn_pos = get_wordnet_pos(treebank_pos)
        if wn_pos is None:
            continue                               # skip unrecognised POS

        p, n, _ = get_sentiment_score(word, wn_pos)
        if p == 0 and n == 0:
            continue                               # fully objective word

        if is_negated:
            p, n = n, p                            # flip scores on negation

        total_pos += p
        total_neg += n
        scored_words += 1

    if scored_words == 0:
        return {"label": "Neutral", "net_score": 0.0,
                "pos_score": 0.0, "neg_score": 0.0}

    # Normalise by number of scored words
    avg_pos = total_pos / scored_words
    avg_neg = total_neg / scored_words
    net     = avg_pos - avg_neg

    if net > threshold:
        label = "Positive"
    elif net < -threshold:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label":     label,
        "net_score": round(net, 4),
        "pos_score": round(avg_pos, 4),
        "neg_score": round(avg_neg, 4),
    }

# ─────────────────────────────────────────────
# 5. BATCH CLASSIFICATION
# ─────────────────────────────────────────────
def classify_dataframe(df, text_column="text", threshold=0.05):
    """
    Apply the classifier to every row in a DataFrame.
    Adds columns: swn_label, swn_net_score, swn_pos_score, swn_neg_score
    """
    results = df[text_column].apply(
        lambda t: sentiwordnet_classify(str(t), threshold)
    )
    df = df.copy()
    df["swn_label"]     = results.apply(lambda r: r["label"])
    df["swn_net_score"] = results.apply(lambda r: r["net_score"])
    df["swn_pos_score"] = results.apply(lambda r: r["pos_score"])
    df["swn_neg_score"] = results.apply(lambda r: r["neg_score"])
    return df

# ─────────────────────────────────────────────
# 6. EVALUATION HELPER
# ─────────────────────────────────────────────
def evaluate(df, true_col="ground_truth_label", pred_col="swn_label"):
    """Print a simple classification report."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("=== SentiWordNet Classifier — Evaluation ===\n")
    print(classification_report(df[true_col], df[pred_col]))
    print("Confusion Matrix:")
    labels = sorted(df[true_col].unique())
    cm = confusion_matrix(df[true_col], df[pred_col], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

# ─────────────────────────────────────────────
# 7. QUICK DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this product, it's amazing!",
        "This is the worst experience I have ever had.",
        "The package arrived on time.",                        # neutral
        "I do not like this at all.",                         # negation
        "Not bad, actually pretty good for the price.",       # double negation
        "Terrible quality, completely disappointed.",
        "It works fine, nothing special.",
    ]

    print(f"{'Text':<50} {'Label':<10} {'Net':>7}  {'Pos':>7}  {'Neg':>7}")
    print("-" * 85)
    for text in sample_texts:
        result = sentiwordnet_classify(text)
        print(f"{text[:49]:<50} {result['label']:<10} "
              f"{result['net_score']:>7.4f}  {result['pos_score']:>7.4f}  "
              f"{result['neg_score']:>7.4f}")

    # ── DataFrame example ──────────────────────────────────────────────
    print("\n--- DataFrame batch example ---")
    df = pd.DataFrame({"text": sample_texts})
    df = classify_dataframe(df)
    print(df[["text", "swn_label", "swn_net_score"]].to_string(index=False))