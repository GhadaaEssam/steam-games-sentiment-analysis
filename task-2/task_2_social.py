import pandas as pd
import argparse
import re
from textblob import Word, TextBlob
from symspellpy.symspellpy import SymSpell, Verbosity
from tqdm import tqdm
import emoji
from googletrans import Translator

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
translator = Translator()

tqdm.pandas()

# run command: python task_2_social.py --input STEAM_GAMES.csv --output STEAM_GAMES_CLEAN.csv --translate --fix_encoding --remove_noise --lowercase --remove_numbers --fix_spelling --lemmatize --extract_tags --remove_stopwords --remove_emojis
# ---------------------------
# Initialize SymSpell
# ---------------------------
sym_spell = SymSpell(max_dictionary_edit_distance=2)
import pkg_resources
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, 0, 1)

# ---------------------------
# CLEANING FUNCTIONS
# ---------------------------

def translate_to_english(text):
    if pd.isna(text) or text.strip() == "":
        return text
    try:
        result = translator.translate(text, dest='en')
        return result.text
    except Exception:
        return text  # return original if translation fails

def fix_encoding(text):
    if pd.isna(text):
        return text
    return text.encode("ascii", "ignore").decode()

def remove_noise(text):
    if pd.isna(text):
        return text
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.strip()

def lowercase_text(text):
    if pd.isna(text):
        return text
    return text.lower()

def remove_numbers(text):
    if pd.isna(text):
        return text
    return re.sub(r"\d+", "", text)

def fix_spelling(text):
    if pd.isna(text):
        return text
    words = text.split()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            # fallback to TextBlob correction
            corrected.append(str(TextBlob(word).correct()))
    return " ".join(corrected)

def lemmatize_text(text):
    if pd.isna(text):
        return text
    words = text.split()
    lemmas = [Word(w).lemmatize() for w in words]
    return " ".join(lemmas)

def remove_stopwords(text):
    if pd.isna(text):
        return text
    # Keep opinion-affecting words
    opinion_words = {'no', 'not', 'nor', 'never', 'neither', 'nobody', 'nothing', 
                     'nowhere', 'hardly', 'scarcely', 'barely', "don't", "doesn't", 
                     "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "isn't", 
                     "aren't", "wasn't", "weren't"}
    stop_words = set(stopwords.words('english')) - opinion_words
    words = text.split()
    return " ".join([w for w in words if w.lower() not in stop_words])

def remove_emojis(text):
    if pd.isna(text):
        return text
    return emoji.replace_emoji(text, replace='').strip()

def drop_empty_after_cleaning(df, text_col):
    before = len(df)
    
    # Convert empty strings to NaN
    df[text_col] = df[text_col].replace(r'^\s*$', pd.NA, regex=True)
    
    # Drop actual NaN rows
    df = df.dropna(subset=[text_col])
    
    after = len(df)
    print(f"Dropped {before - after} empty/null rows after cleaning")
    return df

# ---------------------------
# CATEGORY EXTRACTION
# ---------------------------

def extract_genre(text):
    if pd.isna(text):
        return "Unknown"
    match = re.search(r"'(.*?)'", text)
    if match:
        return match.group(1)
    return "Unknown"

# ---------------------------
# PIPELINE
# ---------------------------

def preprocess(df, args, text_col, category_col):

    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    if args.translate:
        print("Translating to English...")
        df[text_col] = df[text_col].progress_apply(translate_to_english)


    if args.fix_encoding:
        print("Fixing encoding artifacts...")
        df[text_col] = df[text_col].progress_apply(fix_encoding)

    if args.remove_noise:
        print("Removing noise...")
        df[text_col] = df[text_col].progress_apply(remove_noise)

    if args.lowercase:
        print("Converting to lowercase...")
        df[text_col] = df[text_col].progress_apply(lowercase_text)

    if args.remove_numbers:
        print("Removing numbers...")
        df[text_col] = df[text_col].progress_apply(remove_numbers)

    if args.fix_spelling:
        print("Fixing spelling...")
        df[text_col] = df[text_col].progress_apply(fix_spelling)

    if args.lemmatize:
        print("Applying lemmatization...")
        df[text_col] = df[text_col].progress_apply(lemmatize_text)

    if args.extract_tags:
        print("Extracting category tags...")
        df['category'] = df[category_col].apply(extract_genre)

    if args.remove_stopwords:
        print("Removing stopwords...")
        df[text_col] = df[text_col].progress_apply(remove_stopwords)

    if args.remove_emojis:
        print("Removing emojis...")
        df[text_col] = df[text_col].progress_apply(remove_emojis)
    

    # Summary
    print("\nPipeline Summary:")
    print(f"Empty {text_col}: {df[text_col].isna().sum()}")
    if args.extract_tags:
        print(f"Unique categories: {df['category'].nunique()}")
    
    print("Dropping empty/null reviews after cleaning...")
    df = drop_empty_after_cleaning(df, text_col)
    
    return df


# ---------------------------
# MAIN
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Professional Text Preprocessing Pipeline")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--text_column", type=str, default="review_text", help="Text column to clean")
    parser.add_argument("--category_column", type=str, default="genres", help="Column for categories")

    parser.add_argument("--translate", action="store_true") 
    parser.add_argument("--fix_encoding", action="store_true")
    parser.add_argument("--remove_noise", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_numbers", action="store_true")
    parser.add_argument("--fix_spelling", action="store_true")
    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--extract_tags", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true")
    parser.add_argument("--remove_emojis", action="store_true")

    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.input)

    df_clean = preprocess(df, args, args.text_column, args.category_column)

    print("Saving cleaned dataset...")
    df_clean.to_csv(args.output, index=False)

    print("Done!")

if __name__ == "__main__":
    main()