"""
INF7370 - Automne 2025
Émilie Roy
Remise 28 octobre 2025
TP1 Classification sentimentale de tweets

script 1/2
Préparation des données
"""

from nltk.corpus import opinion_lexicon
from nltk.tokenize import TweetTokenizer
from afinn import Afinn
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants
import re

# Changing directory to folder with script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


##### Prepare tools and lexicons #####

# Tokenizer
tknzr = TweetTokenizer()

# Stopwords
stopwords = []
with open("stopwords.txt", "r") as f:
    stopwords = [line.strip() for line in f]

# Emoji / emoticons
with open("emojis_emoticons.txt", "r", encoding="utf-8") as f:
    lines = [line.strip().lower() for line in f if line.strip()] #lowercase to match lowercase tweets
pos_index = lines.index("positive")
neg_index = lines.index("negative")
pos_emos = set(lines[pos_index+1 : neg_index])
neg_emos = set(lines[neg_index+1 :])

# Opinion lexicon sets
pos_words = set(opinion_lexicon.positive())
neg_words = set(opinion_lexicon.negative())

# Sentiment analyzers
afinn = Afinn()
sia = SentimentIntensityAnalyzer()
vader = VaderConstants()


##### Helper functions #####

def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_stopwords(text):
    tokens = tknzr.tokenize(text)
    return [w for w in tokens if w not in stopwords]

def tokens_to_string(tokens):
    return " ".join(tokens)

def count_matches(text, emoticon_set):
    return sum(text.count(e) for e in emoticon_set if e in text)

def process_chunk(df):
    # Preprocessing
    df["cleanText"] = df["SentimentText"].str.lower().apply(remove_urls).apply(remove_stopwords)
    
    # Feature extraction
    df['afinn'] = df['cleanText'].apply(lambda tokens: afinn.score(tokens_to_string(tokens)))
    df['nltk'] = df['cleanText'].apply(lambda tokens: sia.polarity_scores(tokens_to_string(tokens))['compound'])
    df['nbPosWords'] = df['cleanText'].apply(lambda tokens: sum(1 for t in tokens if t in pos_words))
    df['nbNegWords'] = df['cleanText'].apply(lambda tokens: sum(1 for t in tokens if t in neg_words))
    df['negations'] = df['cleanText'].apply(lambda tokens: vader.negated(tokens))
    df['nbPosEmojis'] = df['cleanText'].apply(lambda tokens: count_matches(tokens_to_string(tokens), pos_emos))
    df['nbNegEmojis'] = df['cleanText'].apply(lambda tokens: count_matches(tokens_to_string(tokens), neg_emos))
    df['nbPeriods'] = df['cleanText'].apply(lambda tokens: tokens_to_string(tokens).count('.'))
    df['nbExcla'] = df['cleanText'].apply(lambda tokens: tokens_to_string(tokens).count('!'))
    df['nbInterog'] = df['cleanText'].apply(lambda tokens: tokens_to_string(tokens).count('?'))
    return df


##### Process data in chunks #####

# Adjust processing parameters
chunksize = 10000  # depends on memory
input_csv = "train.csv"
output_csv = "training_data.csv"

# Processing
first_chunk = True
for chunk in pd.read_csv(input_csv, encoding="latin1", chunksize=chunksize):
    chunk = process_chunk(chunk)
    chunk.to_csv(output_csv, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
    first_chunk = False
