import re
import string
import nltk
import pandas as pd
import num2words
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder

fdist = nltk.FreqDist()
nltk.download('stopwords')
nltk.download('punkt')


def import_data(paths: list) -> pd.DataFrame:
    """Import data from a list of paths.
    Args:
        paths (list): List of paths to data files.
    Returns:
        pd.DataFrame: Dataframe with all data.
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path, skipinitialspace=True)
        dfs.append(df)
        print(f"Imported {path}.")
        print(f"Shape: {df.shape}")

    df = pd.concat(dfs)
    print("Done importing data.")
    print(f"Shape: {df.shape}")
    return df


def check_data(df: pd.DataFrame) -> None:
    """Check data.
    Args:
        df (pd.DataFrame): Dataframe with all data.
    Returns:
        None
    """
    print("Checking data...")
    print(f"Shape:\n{df.shape}\n")
    print(f"Columns:\n{df.columns}\n")
    print(f"Number of unique values:\n{df.nunique()}\n")
    print(f"Number of null values:\n{df.isnull().sum()}\n")
    print(f"Number of duplicated values:\n{df.duplicated().sum()}\n")
    print("Done checking data.")


def prepare_data(df: pd.DataFrame, positive_key: str, negative_key: str, date_key: str) -> pd.DataFrame:
    """Clean data.
    Args:
        df (pd.DataFrame): Dataframe with all data.
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """

    reviews = []

    for _, review in df.iterrows():
        review_positive = {
            'review': review[positive_key], 'date': review[date_key], 'score': review['score'], 'target': 'positive'}

        if _check_nulls(review_positive['review']) and _check_length(review_positive['review']) and float(review['score']) >= 5.0:
            reviews.append(review_positive)

        review_negative = {
            'review': review[negative_key], 'date': review[date_key], 'score': review['score'], 'target': 'negative'}

        if _check_nulls(review_negative['review']) and _check_length(review_positive['review']) and float(review['score']) < 5.0:
            reviews.append(review_negative)

    print("Done cleaning data 🎉.")

    df = pd.DataFrame(reviews)

    print(f"Shape: {df.shape}")

    return df


def clean_data(df: pd.DataFrame) -> tuple:
    """Prepare data by removing punctuations and stop words.
    Args:
        df (pd.DataFrame): Dataframe with all data.
    Returns:
        tuple: Tuple of X and y.
    """
    print("Preparing data...")

    # Removing emojis
    df = df.astype(str).apply(lambda x: x.str.encode(
        'ascii', 'ignore').str.decode('ascii'))

    # Remove punctuations
    df['review'] = df['review'].str.replace('[^\w\s]', '')

    # Remove stop words & convert numbers to words
    stop = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda words: ' '.join(
        word.lower() for word in words.split() if word not in stop))

    print("Done preparing data.")

    fdist_sorted = sorted(fdist.items(), key=lambda x: x[1], reverse=True)
    print(fdist_sorted)

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encode labels.
    Args:
        df (pd.DataFrame): Dataframe with all data.
    Returns:
        pd.DataFrame: Dataframe with encoded labels.
    """
    print("Encoding labels...")
    labelEncoder = LabelEncoder()
    df.target = labelEncoder.fit_transform(df.target)
    print("Done encoding labels.")
    return df


# these null equivilant values were seen in the data
null_values = ['null', 'none', 'nothing', 'nil', '.', 'nan',
               'na', 'none as such', 'nothing!', 'no thing', 'no comments']


def _check_nulls(value: str):
    return value != None and str(value).lower() not in null_values and bool(value.strip())


def _check_length(value: str):
    return len(str(value)) > 10
