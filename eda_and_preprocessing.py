import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- NLTK Setup ---
try:
    stopwords.words('english')
    nltk.data.find('corpora/wordnet.zip/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded.")

# --- Configuration ---
INPUT_FILE = 'Resume.csv'
CLEANED_FILE = 'cleaned_resumes.csv'
TRAIN_FILE = 'train.csv'
VALIDATION_FILE = 'validation.csv'
TEST_FILE = 'test.csv'
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25 

# --- Text Preprocessing Function (IMPROVED) ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 2. Anonymize/Remove emails, phone numbers, and URLs
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Remove special characters but keep numbers and key symbols (+, #, .)
    # This preserves terms like 'c++', '.net', 'python 3.9', etc.
    text = re.sub(r'[^a-z0-9\s\+#\.]', '', text)

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Remove stop words and short tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# --- EDA Plotting Functions ---

def plot_category_distribution(df, column='Category'):
    plt.figure(figsize=(12, 8))
    sns.countplot(y=df[column], order=df[column].value_counts().index, palette='viridis')
    plt.title('Distribution of Resume Categories', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.close()
    print("Saved category distribution plot to category_distribution.png")

def plot_word_count_distribution(df, text_col='cleaned_resume', category_col='Category'):
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12, 8))
    sns.boxplot(y=df[category_col], x=df['word_count'], order=df.groupby(category_col)['word_count'].median().sort_values(ascending=False).index, palette='plasma')
    plt.title('Word Count Distribution by Category', fontsize=16)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.xlim(0, df['word_count'].quantile(0.99))
    plt.tight_layout()
    plt.savefig('word_count_distribution.png')
    plt.close()
    print("Saved word count distribution plot to word_count_distribution.png")

def plot_top_n_words(df, text_col='cleaned_resume', n=20):
    all_words = ' '.join(df[text_col]).split()
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(n)
    top_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Count', y='Word', data=top_df, palette='magma')
    plt.title(f'Top {n} Most Common Words in All Resumes', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Word', fontsize=12)
    plt.tight_layout()
    plt.savefig('top_n_words.png')
    plt.close()
    print(f"Saved top {n} words plot to top_n_words.png")


# --- Main Execution ---

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = df[['Resume_str', 'Category']].copy()
    df.rename(columns={'Resume_str': 'original_resume'}, inplace=True)
    df.dropna(subset=['original_resume', 'Category'], inplace=True)
    df.drop_duplicates(inplace=True)

    print("\nStarting text preprocessing with IMPROVED logic...")
    df['cleaned_resume'] = df['original_resume'].apply(preprocess_text)
    print("Text preprocessing complete.")
    df.to_csv(CLEANED_FILE, index=False)
    print(f"Cleaned data saved to {CLEANED_FILE}")

    print("\nStarting Exploratory Data Analysis...")
    plot_category_distribution(df)
    plot_word_count_distribution(df)
    plot_top_n_words(df)
    print("EDA complete.")

    print("\nSplitting data into train, validation, and test sets...")
    final_df = df[['cleaned_resume', 'Category']].copy()
    
    train_val_df, test_df = train_test_split(
        final_df, test_size=TEST_SIZE, random_state=42, stratify=final_df['Category'])

    train_df, val_df = train_test_split(
        train_val_df, test_size=VALIDATION_SIZE, random_state=42, stratify=train_val_df['Category'])
    
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    print("\nData splitting complete. Files saved.")

if __name__ == '__main__':
    main()

