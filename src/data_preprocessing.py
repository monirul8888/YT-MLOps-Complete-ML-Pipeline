import os
import logging
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string
from sklearn.preprocessing import LabelEncoder
import pandas as pd


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(file_handler_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter(" %(asctime)s --- %(name)s --- %(levelname)s --- %(message)s" )
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def transform_text(text):

    """
    This function cleans and transforms text.

    Steps:
    1. Converts text to lowercase
    2. Splits text into words
    3. Removes punctuation and stopwords
    4. Converts words into root form using stemming
    5. Returns the cleaned text

    """

    # Create stemmer object
    ps = PorterStemmer()

    # Convert text to lowercase
    text = text.lower()

    # Split text into words
    words = nltk.word_tokenize(text)

    # Empty list to store cleaned words
    clean_words = []

    # Remove punctuation and stopwords
    for word in words:

        # Keep only letters and numbers
        if word.isalnum():

            # Remove common words like "the", "is", "and"
            if word not in stopwords.words('english'):

                clean_words.append(word)

    # Stem the words
    stemmed_words = []

    for word in clean_words:
        stemmed_words.append(ps.stem(word))

    # Join words into one sentence
    return " ".join(stemmed_words)


 




def preprocess_df(df, text_column='text', target_column='target'):
    """
    This function preprocesses the DataFrame.

    Steps:
    1. Encode target column into numbers
    2. Remove duplicate rows
    3. Clean text data
    4. Return processed DataFrame
    """

    try:
        logger.info("Starting data preprocessing")

        # convert target labels into numbers
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])

        logger.info("Target column encoded")

        # remove duplicate rows
        df = df.drop_duplicates(keep='first')

        logger.info("Duplicate rows removed")

        # clean text column
        df[text_column] = df[text_column].apply(transform_text)

        logger.info("Text column transformed")

        # return cleaned dataframe
        return df

    except KeyError as e:
        logger.error(f"Column not found: {e}")
        raise

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise



def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()