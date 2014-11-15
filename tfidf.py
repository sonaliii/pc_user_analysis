from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



class Tfidf(object):
    def __init__(self):
        self.other_stop = ['lol', 'wtf', 'haha', 'hahaha', 'hahahaha', 'aww', 
                   'awww', 'awwww', 'awwwww', 'omg', 'lmao', 'picture', 
                   'pic', 'photo', 'oh', 'yes', 'no', 'like', 'likes', 
                   'look', 'looks', 'hahah', 'hahahah', 'hahahahah',
                   'thanks', 'thank', 'you', 'ha', 'ah', 'please', 
                   'wow', 'great', 'good', 'awesome', 'go', 'got', 'get',
                   'yup', 'yep', 'yeah', 'really', 'one', 'think', 'hi',
                   'hahahahaha', 'aw', 'so', 'soo', 'sooo', 'soooo',
                   'tho', 'though', 'two', 'didn', 're', 've', 'way',
                   'time', 'best', 'would', 'trying', 'room', 'day',
                   'see', 'gotta', 'im', 'dat', 'hey', 'bae', 'much',
                   'back', 'isn', 'ya', 'let', 'first', 'take', 'us',
                   'come', 'doe', 'pre', 'took', 'taking', 'ur']
    
    def lower_case(self, text):
        return text.lower().strip().strip('\n')

    def remove_punc(self, text):
        exclude = set(string.punctuation)
        text = ''.join(ch for ch in text if ch not in exclude)
        return text
        
    def remove_unicode(self, text):
        text = ''.join([char if ord(char) < 128 else ' ' for char in text])
        return text

    def stop_removal(self, text):
        stop = stopwords.words('english')
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in stop]

    def stemmer(self, text):
        snowball = SnowballStemmer('english')
        snowball_text = []
        for article in text:
            snowball_tokens = []
            for token in article:
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



# articles_counts = []
# for article in snowball_articles:
#     article_counts = []
#     for word in vocab:
#         article_counts.append(article.count(word))
#     articles_counts.append(article_counts)

processed_articles = []

for article in list_of_articles:
    lowered = lower_case(article)
    punc_removed = remove_punc(lowered)
    unicode_removed = remove_unicode(punc_removed)
    stop_removed = stop_removal(unicode_removed)
    processed_articles.append(stop_removed)    

tf = TfidfVectorizer(input = 'content', stop_words = 'english')
tf_matrix = tf.fit_transform(processed_articles)
