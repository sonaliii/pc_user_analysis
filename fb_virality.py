import ast
import base64
import json
from collections import defaultdict

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
from sklearn.metrics import silhouette_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from scipy.spatial.distance import pdist, squareform
from nltk.corpus import stopwords
import matplotlib
from pylab import *
import seaborn

from referrals import Referrals


class Facebook(object):
    def __init__(self):
        self.filenames = ['../data/fb.json', '../data/fb2.json',
                          '../data/fb3.json', '../data/fb4.json',
                          '../data/fb5.json', '../data/fb6.json',
                          '../data/fb7.json', '../data/fb8.json',
                          '../data/fb9.json', '../data/fb10.json']
        self.comments = pd.read_csv('../data/comments.csv')
        self.captions = pd.read_csv('../data/captions.csv')
        r = Referrals()
        referral_weeks = r.join_referrals_weeks()
        priority_users = r.priority_by_weeks(referral_weeks)
        self.users = r.count_circles(priority_users)

        self.vocab = []
        self.stop = stopwords.words('english')
        self.other_stop = ['lol', 'wtf', 'haha', 'hahaha', 'hahahaha', 'aww',
                           'awww', 'awwww', 'awwwww', 'omg', 'lmao', 'picture',
                           'pic', 'photo', 'oh', 'yes', 'no', 'like', 'likes',
                           'look', 'looks', 'hahah', 'hahahah', 'hahahahah',
                           'thanks', 'thank', 'you', 'ha', 'ah', 'please',
                           'wow', 'great', 'good', 'awesome', 'go', 'got',
                           'yup', 'yep', 'yeah', 'really', 'one', 'think',
                           'hahahahaha', 'aw', 'so', 'soo', 'sooo', 'soooo',
                           'tho', 'though', 'two', 'didn', 're', 've', 'way',
                           'time', 'best', 'would', 'trying', 'room', 'day',
                           'see', 'gotta', 'im', 'dat', 'hey', 'bae', 'much',
                           'back', 'isn', 'ya', 'let', 'first', 'take', 'us',
                           'come', 'doe', 'pre', 'took', 'taking', 'ur', 'get',
                           'hi']

    def load_fb_data(self, filename):
        '''
        Loads Facebook data from a given file into a pandas DataFrame.
        Replaces NA's with 0 for ID values
        '''
        f = open(filename, 'r')
        data = json.load(f)
        df = pd.DataFrame(data)

        df['id'] = df['id'].fillna(0)
        df = df[df['id'] != 0]
        return df

    @property
    def load_all_fbs(self):
        '''
        Loads all Facebook data from list of filenames
        '''
        dfs = []
        for f in self.filenames:
            df = self.load_fb_data(f)
            dfs.append(df)
        return dfs

    @property
    def merge_dfs(self):
        '''
        Merges all individual Facebook user dataframes into one dataframe
        '''
        dfs = self.load_all_fbs
        final_df = dfs[0]
        for df in dfs[1:]:
            final_df = final_df.append(df)
        fb_users = final_df.drop_duplicates(cols='id')
        return fb_users

    def extract_work_info(self, fb_users):
        '''
        Extracts Facebook user work information from the DataFrame.
        Pulls out the first listed employer and position for each person.
        '''
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
        '''
        Extracts Facebook user hometown information from DataFrame
        Replaces missing hometowns with "None"
        '''
        hometowns = list(fb_users['hometown'].fillna("{'name': 'None'}"))
        homes = []
        for hometown in hometowns:
            if isinstance(hometown, str):
                hometown = ast.literal_eval(hometown)
            homes.append(hometown['name'])
        fb_users['hometowns'] = homes
        return fb_users

    @property
    def combine_all_features(self):
        '''
        Combines all features from Facebook dataframes
        and selects final columns to use
        '''
        df = self.merge_dfs
        fb_users = self.users
        fb_users = fb_users.merge(df, how='right',
                                  left_on='FacebookID',
                                  right_on='id')
        fb_users['name'] = fb_users['name'].fillna(-1)

        #Selecting columns/features to return
        user_cols = list(self.users.columns)
        extra_cols = ['gender', 'name', 'locale']
        return fb_users[user_cols + extra_cols]

    @property
    def dummy_gender(self):
        '''
        Create dummy variable out of gender (1 female, 0 male, -1 unidentified)
        '''
        fb_users = self.combine_all_features
        fb_users['gender'] = fb_users['gender'].apply(lambda x: 1 if x == 'female' else 0 if x == 'male' else -1)
        return fb_users

    @property
    def only_users_with_gender(self):
        '''
        Select only users from the DataFrame with a given feature
        '''
        fb_users = self.dummy_gender
        fb_users = fb_users[fb_users['gender'] != -1]
        return fb_users

    @property
    def binarize_locale(self):
        '''
        Binarize any given column to be used in machine learning model
        '''
        #Preparing and cleaning data for machine learning
        fb_users = self.only_users_with_gender
        binarized = pd.get_dummies(fb_users['locale'])
        return binarized

    @property
    def group_locales(self):
        '''
        Group locales with fewer than 100 users
        '''
        locales = self.binarize_locale
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

    def create_X(self, fb_users, locales):
        '''
        Creates X dataframe for model
        '''
        X = fb_users[['gender']]
        X = X.join(locales)
        # X = X.merge(tftransformed, how='inner', left_on='UserID', right_on='UserID')
        # X['Circles'] = X['Circles'].fillna(0)
        # X = X.drop('UserID', axis=1)
        return X

    def create_y(self, fb_users):
        '''
        Creates y vector for model
        '''
        y = fb_users[['Priority']].fillna(0)
        # y = y.merge(tftransformed, how='inner', left_on='UserID', right_on='UserID')
        return np.array(y).ravel()

    def grid_search(self, X, y):
        '''
        Grid Search for the best parameters for the given model
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        rf = RandomForestClassifier()
        gs = GridSearchCV(rf, param_grid={'max_depth': [1, 2, 3, 4, 5, None],
                          'n_estimators': [20, 50, 100, 120], 'n_jobs': [-1]})
        gs.fit(X, y)
        return gs.best_params_

    def kfold_cv(self, X, y):
        '''
        KFold Cross Validation for Random Forest and Logistic Regression
        '''
        X = np.array(X)
        kf = KFold(X.shape[0], n_folds=4)
        scores = []
        lr_scores = []

        for train_index, test_index in kf:
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index]
            rf = RandomForestClassifier(max_depth=3, n_estimators=100, n_jobs=-1)
            model = rf.fit_transform(X_train, y_train)
            rf_pred = rf.predict(X_test)
            print confusion_matrix(y_test, rf_pred), 'random forest confusion matrix'
            scores.append(rf.score(X_test, y_test))
            lr = LogisticRegression()
            model = lr.fit_transform(X_train, y_train)
            lr_pred = lr.predict(X_test)
            print confusion_matrix(y_test, lr_pred), 'logistic regression confusion matrix'
            lr_scores.append(lr.score(X_test, y_test))

        return np.mean(scores), np.mean(lr_scores)

    def build_rf(self, X, y):
        '''
        Builds random forest model
        '''
        #Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        rf = RandomForestClassifier(max_depth=3, n_estimators=100, n_jobs=-1)
        model = rf.fit_transform(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_probs = rf.predict_proba(X_test)[:, 1]
        # print y_probs.describe()
        # print y_probs.info()
        print X.columns, 'X columns'
        print np.mean(y_probs), 'mean probs'
        y_probs = y_probs >= 0.054
        print confusion_matrix(y_test, y_pred)
        return rf, rf.score(X_test, y_test)

    def find_important_features(self, rf):
        '''
        Selects the top 20 most important features from the random forest classifier
        '''
        feature_indices = np.argsort(rf.feature_importances_)[::-1][:20]
        important_cols = X.columns[feature_indices]
        return important_cols

    def priority_ratios(self, X, y):
        '''
        Priority ratios (how many people from each category are considered "Priority" users)
        '''
        ratios = []
        for col in X.columns:
            conditions = X[col] == 1
            printout = col + '\t' + str(np.mean(y['Priority'][conditions])) + '\t\t\t\t(' + str(np.mean(y['Priority'][conditions]) * sum(X[col][conditions])) + '/' + str(sum(X[col][conditions])) + ')'
            print printout
            ratios.append(printout)
        print 'male\t' + str(np.mean(y['Priority'][X['gender'] == 0]))
        return ratios

    def plot_priority_ratios(self, X, y):
        '''
        Plotting priority ratios
        '''
        for i, col in enumerate(X.columns):
            priority_circles = mean(X[col][y['Priority'] == True])
            all_circles = mean(X[col][y['Priority'] == False])
            figsize(3, len(X.columns) * 4)
            subplot(len(X.columns), 1, i)

            plt.bar([0, 1], [all_circles, priority_circles])
            plt.xticks([0.4, 1.4], ('Low Priority', 'High Priority'))
            plt.ylabel(col)
            plt.title(col + ' by Low and High Priority Users')

    def tfidf_comments(self, fb_users):
        '''
        TF-IDF analysis on photo comment text
        '''
        comments = self.comments
        decoded_ids = []
        for uid in comments['user_id']:
            decoded_ids.append(base64.b64decode(uid))
        comments['user_id'] = decoded_ids

        comments = comments.merge(fb_users, how='left',
                                  left_on='user_id',
                                  right_on='UserID')

        stop = self.stop + self.other_stop
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                     stop_words=stop,
                                     max_features=100)
        model = vectorizer.fit_transform(comments['comment_text'])
        model = pd.DataFrame(model.todense(), columns=vectorizer.get_feature_names())

        model = model.set_index(comments['user_id'])
        model = model.reset_index()
        return vectorizer, model

    def tfidf_captions(self, fb_users):
        '''
        TF-IDF analysis on photo caption text
        '''
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
        '''
        Non-negative matrix factorization for TF-IDF transformed text
        '''
        # transformed = transformed.drop('user_id', axis=1)
        transformed = transformed.drop('UserID', axis=1)

        #Non-negative matrix factorization (clustering)
        nmf = NMF(n_components=20).fit(transformed)
        feature_names = tf.get_feature_names()
        self.vocab = []
        for topic_idx, topic in enumerate(nmf.components_):
            self.vocab.append(feature_names[i] for i in topic.argsort()[:-11:-1])
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-11:-1]]))
            print()
        self.vocab = list(set(self.vocab))
        return nmf.reconstruction_err_

    def tfidf_limited_vocab(self, fb_users):
        '''
        Uses only a limited vocabulary of top most important text features for TF-IDF
        '''
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
        '''
        KMeans clustering of TF-IDF transformed text
        '''
        #KMeans (also clustering)
        transformed = transformed.drop('UserID', axis=1)
        # kmean = kmeans_model.fit_transform(transformed)
        kmeans_model = KMeans(n_clusters=10, random_state=1).fit(transformed)
        kmean = kmeans_model.transform(transformed)
        labels = kmeans_model.labels_
        # silhouette = silhouette_score(transformed, labels, metric='euclidean')
        centroids = kmeans_model.cluster_centers_

        #Find top 10 words for each cluster
        centroid_top_indices = []
        for centroid in centroids:
            centroid_sorted = np.array(centroid).argsort()[::-1]
            centroid_top_indices.append(centroid_sorted[0:10])

        features = np.array(tf.get_feature_names())
        centroid_top_feats = []
        for centroid in centroid_top_indices:
            centroid_top_feats.append(features[centroid])

        cluster_assignments = kmeans_model.fit_predict(transformed)

        clusters = defaultdict(list)
        for i, j in enumerate(cluster_assignments[0:32]):
            clusters[j].append(i)

        # for cluster in dict(clusters).iteritems():
        #     print cluster
        #     for item in cluster:
        #         print self.comments['comment_text'][item]
        return centroid_top_feats, labels

    def silhouette(self, transformed, labels):
        """
        Computes the silhouette score for each instance of a clustered dataset,
        which is defined as:
            s(i) = (b(i)-a(i)) / max{a(i),b(i)}
        with:
            -1 <= s(i) <= 1

        Args:
            X    : A M-by-N array of M observations in N dimensions
            cIDX : array of len M containing cluster indices (starting from zero)

        Returns:
            s    : silhouette value of each observation
        """
        transformed = transformed.drop('UserID', axis=1)
        X = transformed
        cIDX = labels

        N = X.shape[0]              # number of instances
        K = len(np.unique(cIDX))    # number of clusters

        # compute pairwise distance matrix
        D = squareform(pdist(X))

        # indices belonging to each cluster
        kIndices = [np.flatnonzero(cIDX == k) for k in range(K)]

        # compute a,b,s for each instance
        a = np.zeros(N)
        b = np.zeros(N)
        for i in range(N):
            # instances in same cluster other than instance itself
            a[i] = np.mean([D[i][ind] for ind in kIndices[cIDX[i]] if ind != i])
            # instances in other clusters, one cluster at a time
            b[i] = np.min([np.mean(D[i][ind])
                          for k, ind in enumerate(kIndices) if cIDX[i] != k])
        s = (b - a)/np.maximum(a, b)

        return s

if __name__ == '__main__':
    fb = Facebook()
    fb_users = fb.only_users_with_gender
    tf, tmodel = fb.tfidf_comments(fb_users)
    tf2, tmodel2 = fb.tfidf_captions(fb_users)
    # n2 = fb.nm(tf2, tmodel2)
    # tf_ltd, tmodel_ltd = fb.tfidf_limited_vocab(fb_users)
    # X = fb.create_X(fb_users, locales)

    # y = fb.create_y(fb_users)

    # k, labels = fb.km(tf2, tmodel2)
    n = fb.nm(tf2, tmodel2)
    print n, 'reconstruction error'
    # rf_scores, lr_scores = fb.kfold_cv(X, y)
    # print rf_scores, 'RF score'
    # print lr_scores, 'LR score'
    # print fb.grid_search(X, y)
    # model, score = fb.build_rf(X, y)
    # important_cols = fb.find_important_features(model)
    # print important_cols, 'important columns'
    # print score, 'accuracy score'
    # fb.priority_ratios(X, y)
    # print k, 'KMeans'
    # print np.mean(fb.silhouette(tmodel2, labels)), 'mean silhouette score'
