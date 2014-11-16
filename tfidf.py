import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Tfidf(object):
    def __init__(self):
        self.comments = pd.read_csv('../data/comments.csv')
        self.captions = pd.read_csv('../data/captions.csv')
        self.other_stop = ['lol', 'wtf', 'haha', 'hahaha', 'hahahaha', 'aww',
                           'awww', 'awwww', 'awwwww', 'omg', 'lmao', 'picture',
                           'pic', 'photo', 'oh', 'yes', 'no', 'like', 'likes',
                           'look', 'looks', 'hahah', 'hahahah', 'hahahahah',
                           'thanks', 'thank', 'you', 'ha', 'ah', 'please',
                           'wow', 'great', 'good', 'awesome', 'go', 'got',
                           'get', 'yup', 'yep', 'yeah', 'really', 'one',
                           'think', 'hi', 'hahahahaha', 'aw', 'so', 'soo',
                           'sooo', 'soooo', 'tho', 'though', 'two', 'didn',
                           're', 've', 'way', 'gonna', 'time', 'best',
                           'would', 'trying', 'room', 'day', 'see', 'gotta',
                           'im', 'dat', 'hey', 'bae', 'much', 'back', 'isn',
                           'ya', 'let', 'first', 'take', 'us', 'taking',
                           'come', 'doe', 'pre', 'took', 'ur']

    @property
    def lower_case(self):
        lowered = []
        for caption in self.captions['caption']:
            lowered.append(caption.lower().strip().strip('\n'))
        return lowered

    #Computer runs out of memory in remove_punc
    @property
    def remove_punc(self):
        text = self.lower_case
        exclude = set(string.punctuation)
        captions = []
        for caption in text:
            captions.append(''.join(ch for ch in text if ch not in exclude))
        # text = ''.join(ch for ch in text if ch not in exclude)
        return captions

    @property
    def remove_unicode(self):
        text = self.remove_punc
        unicode_removed = []
        for caption in text:
            unicode_removed.append(''.join([char if ord(char) < 128 else ' ' for char in caption]))
        return unicode_removed

    @property
    def stop_removal(self):
        text = self.remove_unicode
        stop = stopwords.words('english') + self.other_stop
        tokenized = []
        for word in text:
            tokenized.append(word_tokenize(word))
        return [word for word in tokenized if word not in stop]

    @property
    def stemmer(self):
        text = self.stop_removal
        snowball = SnowballStemmer('english')
        snowball_text = []
        for caption in text:
            snowball_tokens = []
            for token in caption:
                snowball_tokens.append(snowball.stem(token))
            snowball_text.append(snowball_tokens)
        return snowball_text

    def make_vocab(self, snowball_text):
        vocab = []
        for article in snowball_articles:
            for token in article:
                vocab.append(token)
        vocab = list(set(vocab))
        return vocab

    @property
    def vectorizer(self):
        processed_text = self.stemmer
        print processed_text[4]
        tf = TfidfVectorizer(input='content', max_features=2000)
        tf_matrix = tf.fit_transform(processed_text)
        return tf, tf_matrix


if __name__ == '__main__':
    tf = Tfidf()
    # tf.vectorizer()
    print tf.remove_punc[380]
# processed_articles = []

# for article in list_of_articles:
#     lowered = lower_case(article)
#     punc_removed = remove_punc(lowered)
#     unicode_removed = remove_unicode(punc_removed)
#     stop_removed = stop_removal(unicode_removed)
#     processed_articles.append(stop_removed)
