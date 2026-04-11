#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
print(sys.executable)


# In[3]:


get_ipython().system('{sys.executable} -m pip install textblob')


# In[4]:


get_ipython().system('{sys.executable} -m textblob.download_corpora')


# In[9]:


"""
BAG OF WORDS CONVERTER FOR ML TEAM
==================================
Input: 3 CSV files from ML team (Scheme A, B, C)
Output: 3 BoW CSV files ready for Naive Bayes & Decision Tree
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================

# Location of ML team's files
INPUT_PATH = r"C:\Users\USER\Downloads\social"

# Exact filenames from ML team (UPDATE THESE)
SCHEME_A_FILE = "SCHEMA_1.csv"     
SCHEME_B_FILE = "SCHEMA_2.csv"       
SCHEME_C_FILE = "SCHEMA_3.csv"     

# Output filenames
OUTPUT_A = "bow_scheme_a.csv"
OUTPUT_B = "bow_scheme_b.csv"
OUTPUT_C = "bow_scheme_c.csv"

# BoW Parameters
MAX_FEATURES = 5000    # Maximum number of words in vocabulary
MIN_DF = 2            # Minimum document frequency (ignore rare words)
MAX_DF = 0.95         # Maximum document frequency (ignore common words)

# ============================================
# FUNCTION TO APPLY BAG OF WORDS
# ============================================

def apply_bag_of_words(csv_filepath, text_column_name, output_filepath, scheme_name):
    """
    Convert text column to Bag of Words matrix
    
    Parameters:
    - csv_filepath: Path to input CSV file
    - text_column_name: Name of column containing text
    - output_filepath: Where to save BoW CSV
    - scheme_name: Name of scheme (for logging)
    
    Returns:
    - Dictionary with results
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {scheme_name}")
    print(f"{'='*60}")
    
    # Step 1: Load CSV
    print(f"📂 Loading file: {csv_filepath}")
    try:
        df = pd.read_csv(csv_filepath)
        print(f"   ✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"   Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"   ❌ ERROR: File not found!")
        print(f"   Please check path: {csv_filepath}")
        return None
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return None
    
    # Step 2: Identify text column (if not specified)
    if text_column_name is None:
        # Auto-detect: find first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                text_column_name = col
                break
    
    if text_column_name not in df.columns:
        print(f"   ❌ ERROR: Column '{text_column_name}' not found!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    print(f"📝 Using text column: '{text_column_name}'")
    
    # Step 3: Extract text and handle missing values
    texts = df[text_column_name].fillna('').astype(str).tolist()
    
    # Show sample
    print(f"📄 Sample text (first 100 chars):")
    print(f"   '{texts[0][:100]}...'")
    
    # Check for empty texts
    empty_count = sum(1 for t in texts if len(t.strip()) == 0)
    print(f"⚠️  Empty texts found: {empty_count} / {len(texts)}")
    
    if empty_count == len(texts):
        print(f"   ❌ ALL TEXTS ARE EMPTY! Cannot create BoW.")
        return None
    
    # Step 4: Apply Bag of Words
    print(f"🔧 Creating Bag of Words matrix...")
    
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        lowercase=False,  # Text already preprocessed by ML team
        token_pattern=r'(?u)\b\w+\b'  # Token pattern for words
    )
    
    try:
        X = vectorizer.fit_transform(texts)
        print(f"   ✅ BoW created successfully!")
    except Exception as e:
        print(f"   ❌ ERROR during vectorization: {e}")
        return None
    
    # Step 5: Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    bow_df = pd.DataFrame(X.toarray(), columns=feature_names)
    
    # Step 6: Calculate statistics
    total_cells = X.shape[0] * X.shape[1]
    non_zero = X.nnz
    density = (non_zero / total_cells) * 100
    
    # Check for all-zero rows
    rows_all_zero = (X.sum(axis=1) == 0).sum()
    
    print(f"\n📊 STATISTICS:")
    print(f"   - Documents: {X.shape[0]:,}")
    print(f"   - Vocabulary size: {X.shape[1]:,}")
    print(f"   - Non-zero cells: {non_zero:,}")
    print(f"   - Matrix density: {density:.4f}%")
    print(f"   - Rows with all zeros: {rows_all_zero} / {X.shape[0]}")
    print(f"   - Columns (words): {feature_names[:10].tolist()}...")
    
    # Step 7: Save to CSV
    print(f"💾 Saving to: {output_filepath}")
    bow_df.to_csv(output_filepath, index=False)
    print(f"   ✅ Saved successfully!")
    
    return {
        'shape': X.shape,
        'non_zero': non_zero,
        'density': density,
        'empty_rows': rows_all_zero,
        'features': feature_names,
        'dataframe': bow_df
    }

# ============================================
# MAIN EXECUTION
# ============================================

print("\n" + "="*60)
print("BAG OF WORDS CONVERTER")
print("Converting ML Team's Cleaned Text → BoW Matrices")
print("="*60)

# Ask user for column names if unknown
print("\n📌 IMPORTANT: What is the name of the column containing text?")
print("   (e.g., 'review_text', 'clean_text', 'text', etc.)")
text_column = input("   Column name: ").strip()

if not text_column:
    text_column = None  # Will auto-detect
    print("   → Auto-detecting text column...")

# Ask user to confirm filenames
print("\n📌 Please confirm your file names:")
print(f"   Scheme A file (Light): {SCHEME_A_FILE}")
confirm_a = input("   Is this correct? (y/n): ").strip().lower()
if confirm_a == 'n':
    SCHEME_A_FILE = input("   Enter correct filename: ").strip()

print(f"   Scheme B file (Standard): {SCHEME_B_FILE}")
confirm_b = input("   Is this correct? (y/n): ").strip().lower()
if confirm_b == 'n':
    SCHEME_B_FILE = input("   Enter correct filename: ").strip()

print(f"   Scheme C file (Aggressive): {SCHEME_C_FILE}")
confirm_c = input("   Is this correct? (y/n): ").strip().lower()
if confirm_c == 'n':
    SCHEME_C_FILE = input("   Enter correct filename: ").strip()

# Process all three schemes
results = {}

# Scheme A
result_a = apply_bag_of_words(
    csv_filepath=f"{INPUT_PATH}\\{SCHEME_A_FILE}",
    text_column_name=text_column,
    output_filepath=f"{INPUT_PATH}\\{OUTPUT_A}",
    scheme_name="SCHEME A (Light Preprocessing)"
)
if result_a:
    results['A'] = result_a

# Scheme B
result_b = apply_bag_of_words(
    csv_filepath=f"{INPUT_PATH}\\{SCHEME_B_FILE}",
    text_column_name=text_column,
    output_filepath=f"{INPUT_PATH}\\{OUTPUT_B}",
    scheme_name="SCHEME B (Standard Preprocessing)"
)
if result_b:
    results['B'] = result_b

# Scheme C
result_c = apply_bag_of_words(
    csv_filepath=f"{INPUT_PATH}\\{SCHEME_C_FILE}",
    text_column_name=text_column,
    output_filepath=f"{INPUT_PATH}\\{OUTPUT_C}",
    scheme_name="SCHEME C (Aggressive Preprocessing)"
)
if result_c:
    results['C'] = result_c

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

if len(results) == 0:
    print("\n❌ No files were processed successfully!")
    print("\nTroubleshooting:")
    print("1. Check that the CSV files exist in:", INPUT_PATH)
    print("2. Verify the filenames are correct")
    print("3. Make sure the text column name is right")
else:
    print("\n✅ Successfully processed files:")
    for scheme, data in results.items():
        print(f"\n   SCHEME {scheme}:")
        print(f"   - Input: {SCHEME_A_FILE if scheme=='A' else SCHEME_B_FILE if scheme=='B' else SCHEME_C_FILE}")
        print(f"   - Output: {OUTPUT_A if scheme=='A' else OUTPUT_B if scheme=='B' else OUTPUT_C}")
        print(f"   - Size: {data['shape'][0]} documents × {data['shape'][1]} words")
        print(f"   - Density: {data['density']:.4f}%")
        
        if data['empty_rows'] == data['shape'][0]:
            print(f"   ⚠️  WARNING: ALL documents are empty!")
        elif data['empty_rows'] > 0:
            print(f"   ⚠️  {data['empty_rows']} documents have no words")
        else:
            print(f"   ✅ All documents have at least one word")
    
    print("\n" + "="*60)
    print("📤 SEND THESE FILES TO ML TEAM:")
    print("="*60)
    if 'A' in results:
        print(f"   1. {OUTPUT_A} - Bag of Words (Scheme A)")
    if 'B' in results:
        print(f"   2. {OUTPUT_B} - Bag of Words (Scheme B)")
    if 'C' in results:
        print(f"   3. {OUTPUT_C} - Bag of Words (Scheme C)")
    
    print("\n" + "="*60)
    print("💡 INSTRUCTIONS FOR ML TEAM:")
    print("="*60)
    print("""
    Each CSV file contains:
    - Rows: Individual reviews/documents
    - Columns: Unique words from vocabulary
    - Values: Frequency of each word in the document
    
    Usage example:
    
    import pandas as pd
    from sklearn.naive_bayes import MultinomialNB
    
    # Load BoW features
    X = pd.read_csv('bow_scheme_a.csv')
    
    # Load labels (assuming you have them)
    y = pd.read_csv('labels.csv')['sentiment']
    
    # Train model
    model = MultinomialNB()
    model.fit(X, y)
    """)
    
    # Quick verification test
    print("\n" + "="*60)
    print("QUICK VERIFICATION TEST")
    print("="*60)
    
    test_file = f"{INPUT_PATH}\\{OUTPUT_A}"
    try:
        test_df = pd.read_csv(test_file)
        print(f"\n✅ Testing {OUTPUT_A}:")
        print(f"   Shape: {test_df.shape}")
        print(f"   First 5 columns: {test_df.columns[:5].tolist()}")
        print(f"   First row (first 10 values): {test_df.iloc[0, :10].tolist()}")
        print(f"   Data type: {test_df.iloc[0,0].__class__.__name__}")
        
        if isinstance(test_df.iloc[0,0], (int, float, np.integer, np.floating)):
            print("\n   ✅ CORRECT: Cells contain numbers (word frequencies)")
        else:
            print("\n   ❌ ERROR: Cells contain text instead of numbers")
    except:
        pass

print("\n🎉 DONE! Ready to send to ML team.")


# In[ ]:




