import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import argparse
import os

# Download required NLTK resources
nltk.download('sentiwordnet', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ─────────────────────────────────────────────
# 1. POS TAG CONVERSION
# ─────────────────────────────────────────────
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

# ─────────────────────────────────────────────
# 2. SENTIWORDNET SCORE LOOKUP
# ─────────────────────────────────────────────
def get_sentiment_score(word, pos):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos=pos)
    synsets = list(swn.senti_synsets(lemma, pos))
    if not synsets:
        return 0.0, 0.0, 1.0
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
    WINDOW = 3
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
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    token_neg_pairs = apply_negation(tokens)

    total_pos, total_neg = 0.0, 0.0
    scored_words = 0

    for (word, treebank_pos), (_, is_negated) in zip(tagged, token_neg_pairs):
        wn_pos = get_wordnet_pos(treebank_pos)
        if wn_pos is None:
            continue
        p, n, _ = get_sentiment_score(word, wn_pos)
        if p == 0 and n == 0:
            continue
        if is_negated:
            p, n = n, p
        total_pos += p
        total_neg += n
        scored_words += 1

    if scored_words == 0:
        return {"label": "Neutral", "net_score": 0.0,
                "pos_score": 0.0, "neg_score": 0.0}

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
def evaluate(df, true_col, pred_col="swn_label"):
    from sklearn.metrics import classification_report, confusion_matrix

    # Normalize case for both columns
    true = df[true_col].str.lower().str.strip()
    pred = df[pred_col].str.lower().str.strip()

    print("=== SentiWordNet Classifier — Evaluation ===\n")
    print(classification_report(true, pred))
    print("Confusion Matrix:")
    labels = sorted(true.unique())
    cm = confusion_matrix(true, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

# ─────────────────────────────────────────────
# 7. LOAD FILE HELPER
# ─────────────────────────────────────────────
def load_file(filepath):
    """Load CSV or Excel file into a DataFrame based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    elif ext == ".json":
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Use CSV, Excel, or JSON.")

# ─────────────────────────────────────────────
# 8. ARGPARSE ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SentiWordNet Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Input ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=False,
        help="Path to input file (CSV / Excel / JSON). If omitted, runs the built-in demo."
    )
    parser.add_argument(
        "--text-col", "-t",
        type=str,
        default="text",
        help="Name of the column containing the text to classify."
    )

    # ── Output ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the results (CSV). If omitted, prints to console."
    )

    # ── Classifier options ─────────────────────────────────────────────
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Net score threshold for Positive/Negative classification."
    )

    # ── Evaluation ─────────────────────────────────────────────────────
    parser.add_argument(
        "--true-col",
        type=str,
        default=None,
        help="Name of the ground-truth label column for evaluation. Skipped if not provided."
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # ── Demo mode (no input file) ──────────────────────────────────────
    if args.input is None:
        print("No input file provided — running built-in demo.\n")
        sample_texts = [
            "I absolutely love this product, it's amazing!",
            "This is the worst experience I have ever had.",
            "The package arrived on time.",
            "I do not like this at all.",
            "Not bad, actually pretty good for the price.",
            "Terrible quality, completely disappointed.",
            "It works fine, nothing special.",
        ]
        df = pd.DataFrame({"text": sample_texts})

    # ── File mode ──────────────────────────────────────────────────────
    else:
        print(f"Loading file: {args.input}")
        df = load_file(args.input)
        print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}\n")

        if args.text_col not in df.columns:
            raise ValueError(
                f"Column '{args.text_col}' not found. "
                f"Available columns: {list(df.columns)}\n"
                f"Use --text-col to specify the correct column name."
            )

    # ── Classify ───────────────────────────────────────────────────────
    print(f"Classifying using threshold={args.threshold} ...")
    df = classify_dataframe(df, text_column=args.text_col, threshold=args.threshold)

    # ── Print sample results ───────────────────────────────────────────
    print("\n--- Results (first 10 rows) ---")
    print(df[[args.text_col, "swn_label", "swn_net_score"]].head(10).to_string(index=False))

    # ── Evaluate if ground truth provided ─────────────────────────────
    if args.true_col:
        if args.true_col not in df.columns:
            print(f"\nWarning: --true-col '{args.true_col}' not found in DataFrame. Skipping evaluation.")
        else:
            print()
            evaluate(df, true_col=args.true_col)

    # ── Save output ────────────────────────────────────────────────────
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n(Use --output results.csv to save results to a file)")


# To run the script:
# Demo mode (no file needed)
## python SentiWordNetClassifier.py

# With a CSV file
## python SentiWordNetClassifier.py --input data.csv --text-col text

# Custom column name + save output
## python SentiWordNetClassifier.py --input reviews.csv --text-col review_text --output results.csv

# With ground truth evaluation
## python SentiWordNetClassifier.py --input GROUBD_TRUTH_WITH_FINAL_LABEL.CSV --text-col review_text --true-col final_label --output SWresults.csv

# With Excel input + custom threshold
## python SentiWordNetClassifier.py --input data.xlsx --text-col body --threshold 0.1 --output out.csv