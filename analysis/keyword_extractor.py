import nltk
import string
import re

from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
CUSTOM_STOPS = {'hn', 'show', 'ask', 'tell'} #non meaningful words to avoid
ALL_STOPS = stop_words.union(CUSTOM_STOPS) # using all stops to skip non-meaningful words


def extract_keywords(text, top_n=10):
    sentence = text
    word_tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [word.lower() for word in word_tokens if word.lower() not in ALL_STOPS and word.isalpha()] #filter by turning everything into smaller case and no punctuation
    word_counts = Counter(filtered_sentence)
    top_words = word_counts.most_common(top_n)
    return [word for word, count in top_words]

text = "Show HN: I built a modern web framework using Rust and WebAssembly"
print(extract_keywords(text, 5))