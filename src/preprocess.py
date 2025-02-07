import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """Cleans the input text by removing HTML tags, special characters, and stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = word_tokenize(text)  # Tokenize text
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

def preprocess_dataset(file_path, output_path):
    """Loads, cleans, and saves the dataset."""
    df = pd.read_csv(file_path)  # Load dataset
    df['cleaned_review'] = df['review'].apply(clean_text)  # Apply text cleaning
    df.to_csv(output_path, index=False)  # Save processed dataset
    print("Preprocessing complete. Saved to:", output_path)

# Run the preprocessing
if __name__ == "__main__":
    preprocess_dataset("../data/IMDB Dataset.csv", "../data/cleaned_IMDB_Dataset.csv")

