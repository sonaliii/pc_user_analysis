import ast
import base64
import json

import pandas as pd
import numpy as np
import requests
from multiprocessing.dummy import Pool
import facebook
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from collections import defaultdict
from nltk.corpus import stopwords
import matplotlib
from pylab import *
import seaborn

from referrals import Referrals


class Facebook(object):
    def __init__(self):
        self.filenames = ['../data/fb.json', '../data/fb2.json', '../data/fb3.json', 
            '../data/fb4.json', '../data/fb5.json', '../data/fb6.json', 
            '../data/fb7.json', '../data/fb8.json', '../data/fb9.json', 
            '../data/fb10.json']
        self.comments = pd.read_csv('../data/comments.csv')
        self.captions = pd.read_csv('../data/captions.csv')
        r = Referrals()
        # referrers = r.count_referrals()
        # referral_weeks = r.join_referrals_weeks()
        # priority_users = r.priority_by_weeks(referral_weeks)
        priority_retention = r.priority_retention()
        self.users = r.count_circles(priority_retention)
        print np.median(self.users['weeks'])
        self.vocab = []
        self.stop = stopwords.words('english')
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

    def load_fb_data(self, filename):
        f = open(filename, 'r')
        data = json.load(f)
        df = pd.DataFrame(data)

        df['id'] = df['id'].fillna(0)
        df = df[df['id'] != 0]
        return df

    def load_all_fbs(self):
        dfs = []
        for f in self.filenames:
            df = self.load_fb_data(f)
            dfs.append(df)
        return dfs

    def merge_dfs(self, dfs):
        final_df = dfs[0]
        for df in dfs[1:]:
            final_df = final_df.append(df)
        fb_users = final_df.drop_duplicates(cols='id')
        return fb_users

    def extract_work_info(self, fb_users):
        works = fb_users['work'].fillna("[{'position': {'name': 'None'}, 'employer': {'name': 'None'}}]")
        works = list(works)
        job_titles = []
        employers = []
        for work in works:
            job_title = []
            employer = []
            if isinstance(work, str):
                work = ast.literal_eval(work)
            for pos in work:
                if 'position' in pos.keys():
                    job_title.append(pos['position']['name'])
                else:
                    job_title.append('None')
                if pos['employer']:
                    try:
                        employer.append(pos['employer']['name'])
                    except KeyError:
                        employer.append('None')
                else:
                    employer.append('None')
            job_titles.append(job_title[0])
            employers.append(employer[0])
        fb_users['job_titles'] = job_titles
        fb_users['employers'] = employers
        return fb_users

    def extract_hometown_info(self, fb_users):
        hometowns = list(fb_users['hometown'].fillna("{'name': 'None'}"))
        homes = []
        for hometown in hometowns:
            if isinstance(hometown, str):
                hometown = ast.literal_eval(hometown)
            homes.append(hometown['name'])
        fb_users['hometowns'] = homes
        return fb_users

    def combine_all_features(self, df):
        
        fb_users = self.users
        fb_users = fb_users.merge(df, how='right', left_on='FacebookID', right_on='id')
        fb_users['name'] = fb_users['name'].fillna(-1)
        
        #Selecting columns/features to return
        user_cols = list(self.users.columns)
        extra_cols = ['gender', 'name', 'locale']
        return fb_users[user_cols + extra_cols]

    def binarize_gender(self, fb_users):
        fb_users['gender'] = fb_users['gender'].apply(lambda x: 1 if x == 'female' else 0 if x == 'male' else -1)
        return fb_users

    def binarize_col(self, fb_users, col_name):
        
        #Preparing and cleaning data for machine learning
        binarized = pd.get_dummies(fb_users[str(col_name)])
        return binarized

    def group_locales(self, locales):
        others = []
        for col in locales.columns:
            total = np.sum(locales[col][locales[col] == 1])
            if total < 100:
                others.append(col)

        other = np.zeros(len(locales['en_US']))
        for col in others:
            other = other + locales[col]
            locales = locales.drop(col, axis=1)

        locales['other'] = other
        return locales


    def only_users_with_X(self, fb_users, x):
        for feature in x:
            fb_users = fb_users[fb_users[feature] != -1]
        return fb_users

    def create_X(self, fb_users, locales, tftransformed):       
        X = fb_users[['UserID', 'gender']]
        X = X.join(locales)
        # X = X.merge(tftransformed, how='inner', left_on='UserID', right_on='UserID')
        # X = X.join(employers)
        # X = X.join(job_titles, rsuffix='_job')
        # X = X.join(hometowns, rsuffix='_ht')
        # X['Circles'] = X['Circles'].fillna(0)
        X = X.drop('UserID', axis=1)
        # X = X.drop('user_id', axis=1)
        return X

    def create_y(self, fb_users, tftransformed):
        y = fb_users[['UserID', 'Priority']]
        # y = y.merge(tftransformed, how='inner', left_on='UserID', right_on='UserID')
        y = y[['Priority']]
        return y


    def grid_search(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        rf = RandomForestClassifier()
        gs = GridSearchCV(rf, param_grid={'max_depth': [1,2,3,4,5,None], 'n_estimators': [20,50,100,120], 'n_jobs': [-1]})
        gs.fit(X, y)
        return gs.best_params_

    def kfold_cv(self, X, y):
        kf = KFold(X.shape[0], n_folds=4, shuffle=True)
        scores = []
        X = np.array(X)

        for train_index, test_index in kf:
            y_train, y_test = y['Priority'][train_index], y['Priority'][test_index]
            X_train, X_test = X[train_index], X[test_index]
            rf = RandomForestClassifier(max_depth = 3, n_estimators = 100, n_jobs = -1)
            model = rf.fit_transform(X_train, y_train)
            scores.append(rf.score(X_test, y_test))

        return np.mean(scores)

    def build_model(self, X, y):
        #Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        rf = RandomForestClassifier(max_depth = 3, n_estimators = 100, n_jobs = -1)
        model = rf.fit_transform(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_probs = rf.predict_proba(X_test)[:, 1]
        # print y_probs.describe()
        # print y_probs.info()
        y_probs = y_probs >= 0.088
        print confusion_matrix(y_test, y_probs)
        return rf, rf.score(X_test, y_test)

    def find_important_features(self, rf):
        #Identifying top 10 most important features
        feature_indices = np.argsort(rf.feature_importances_)[::-1][:20]
        important_cols = X.columns[feature_indices]
        return important_cols

    def priority_ratios(self, X, y):
        ratios = []
        for col in X.columns:
            conditions = X[col] == 1
            printout = col + '\t' + str(np.mean(y['Priority'][conditions])) + '\t\t\t\t(' + str(np.mean(y['Priority'][conditions]) * sum(X[col][conditions])) + '/' + str(sum(X[col][conditions])) + ')'
            print printout
            ratios.append(printout)
        print 'male\t' + str(np.mean(y['Priority'][X['gender'] == 0])) 
        return ratios

    def plot_priority_ratios(self, X, y):
        for i, col in enumerate(X.columns):
            priority_circles = mean(X[col][y['Priority'] == True])
            all_circles = mean(X[col][y['Priority'] == False])
            figsize(3, len(X.columns) * 4)
            subplot(len(X.columns), 1, i)

            plt.bar([0, 1], [all_circles, priority_circles])
            plt.xticks([0.4, 1.4], ('Low Priority', 'High Priority'))
            plt.ylabel(col)
            plt.title(col + ' by Low and High Priority Users');

    def tfidf_comments(self, fb_users):
        comments = self.comments
        decoded_ids = []
        for uid in comments['user_id']:
            decoded_ids.append(base64.b64decode(uid))
        comments['user_id'] = decoded_ids

        comments = comments.merge(fb_users, how='left', left_on='user_id', right_on='UserID')
        
        stop = self.stop + self.other_stop
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop, max_features=100)
        model = vectorizer.fit_transform(comments['comment_text'])
        model = pd.DataFrame(model.todense(), columns=vectorizer.get_feature_names())

        model = model.set_index(comments['user_id'])
        model = model.reset_index()
        return vectorizer, model


    def tfidf_captions(self, fb_users):
        captions = self.captions
        captions = captions.merge(fb_users, how='left', left_on='user_id', right_on='UserID')
        stop = self.stop + self.other_stop
        vectorizer = TfidfVectorizer(max_df=0.95, stop_words=stop, max_features=2000)
        model = vectorizer.fit_transform(captions['caption'])
        model = pd.DataFrame(model.todense(), columns=vectorizer.get_feature_names())
        model = model.set_index(captions['UserID'])
        model = model.reset_index()
        return vectorizer, model

    def nm(self, tf, transformed):
        # transformed = transformed.drop('user_id', axis=1)
        transformed = transformed.drop('UserID', axis=1)

        #Non-negative matrix factorization (clustering)
        nmf = NMF(n_components=10).fit(transformed)
        feature_names = tf.get_feature_names()
        self.vocab = []
        for topic_idx, topic in enumerate(nmf.components_):
            self.vocab.append(feature_names[i] for i in topic.argsort()[:-11:-1])
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-11:-1]]))
            print()
        self.vocab = list(set(self.vocab))

    def tfidf_limited_vocab(self, fb_users):
        captions = self.captions
        captions = captions.merge(fb_users, how='left', left_on='user_id', right_on='UserID')

        stop = self.stop + self.other_stop

        vectorizer = TfidfVectorizer(max_df=0.95, stop_words=stop, vocabulary=self.vocab)
        model = vectorizer.fit_transform(captions['caption'])
        model = pd.DataFrame(model.todense(), columns=vectorizer.get_feature_names())
        model = model.set_index(captions['UserID'])
        model = model.reset_index()
        return vectorizer, model

    def km(self, tf, transformed):
        #KMeans (also clustering)
        km = KMeans(n_clusters=6)
        kmean = km.fit_transform(transformed)
        centroids = km.cluster_centers_
        
        #Find top 10 words for each cluster
        centroid_top_indices = []
        for centroid in centroids:
            centroid_sorted = np.array(centroid).argsort()[::-1]
            centroid_top_indices.append(centroid_sorted[0:10])
            
        features = np.array(tf.get_feature_names())
        centroid_top_feats = []
        for centroid in centroid_top_indices:
            centroid_top_feats.append(features[centroid])
            
        cluster_assignments = km.fit_predict(transformed)

        clusters = defaultdict(list)
        for i, j in enumerate(cluster_assignments[0:32]):
            clusters[j].append(i) 
            
        # for cluster in dict(clusters).iteritems():
        #     print cluster
        #     for item in cluster:
        #         print self.comments['comment_text'][item]
        return centroid_top_feats



if __name__ == '__main__':
    fb = Facebook()
    dfs = fb.load_all_fbs()
    fb_users = fb.merge_dfs(dfs)

    fb_users = fb.combine_all_features(fb_users) 
    fb_users = fb.binarize_gender(fb_users)
    fb_users = fb.only_users_with_X(fb_users, ['gender'])
    locales = fb.binarize_col(fb_users, 'locale')
    locales = fb.group_locales(locales)

    # # tf, tmodel = fb.tfidf_comments(fb_users)
    tf2, tmodel2 = fb.tfidf_captions(fb_users)
    n2 = fb.nm(tf2, tmodel2)
    tf_ltd, tmodel_ltd = fb.tfidf_limited_vocab(fb_users)
    X = fb.create_X(fb_users, locales, tmodel_ltd)
    # # print X.shape
    y = fb.create_y(fb_users, tmodel_ltd)
    # # print y.shape
    # k = fb.km(tf, tmodel)
    # # n = fb.nm(tf2, tmodel2)
    # # print 'CAPTIONS'
    print fb.kfold_cv(X, y)
    # print fb.grid_search(X, y)
    # model, score = fb.build_model(X, y)
    # # # print 'model built'
    # important_cols = fb.find_important_features(model)
    # print important_cols, 'important columns'
    # print score, 'accuracy score'
    fb.priority_ratios(X, y)
