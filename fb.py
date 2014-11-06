import pandas as pd
import numpy as np
import requests
from multiprocessing.dummy import Pool
import facebook
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from referrals import Referrals
import matplotlib
import seaborn


class Facebook(object):
    def __init__(self):
        self.filenames = ['../data/fb.json', '../data/fb2.json', '../data/fb3.json', '../data/fb4.json', '../data/fb5.json', '../data/fb6.json', '../data/fb7.json']
        r = Referrals()
        referrers = r.count_referrals()
        priority_users = r.identify_priority_users(referrers)
        self.users = r.count_circles(priority_users)
        print self.users.columns

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

    def combine_all_features(self, df):
        
        fb_users = self.users
        fb_users = fb_users.merge(df, how='right', left_on='FacebookID', right_on='id')
        fb_users['name'] = fb_users['name'].fillna(-1)
        
        #Selecting columns/features to return
        user_cols = list(self.users.columns)
        extra_cols = ['gender', 'name', 'locale']
        print user_cols, 'should include priority'
        return fb_users[user_cols + extra_cols]

    def binarize_gender(self, fb_users):
        fb_users['gender'] = fb_users['gender'].apply(lambda x: 1 if x == 'female' else 0 if x == 'male' else -1)
        return fb_users

    def binarize_locales(self, fb_users):
        
        #Preparing and cleaning data for machine learning
        locales = pd.get_dummies(fb_users['locale'])
        return locales

    def only_users_with_X(self, fb_users, x):
        for feature in x:
            fb_users = fb_users[fb_users[feature] != -1]
        return fb_users

    def create_X(self, fb_users, locales):       
        X = fb_users[['Circles', 'gender']]
        X = X.join(locales)
        X['Circles'] = X['Circles'].fillna(0)
        return X

    def create_y(self, fb_users):
        y = fb_users[['Priority']]
        return y

    def build_model(self, X, y):
        #Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        rf = RandomForestClassifier(max_depth = 3, n_estimators = 100)
        model = rf.fit_transform(X_train, y_train)
        return rf, rf.score(X_test, y_test)

    def find_important_features(self, rf):
        #Identifying top 10 most important features
        feature_indices = np.argsort(rf.feature_importances_)[::-1][:10]
        important_cols = X.columns[feature_indices]
        return important_cols

    def priority_ratios(self, X, y):
        ratios = []
        for col in X.columns:
            conditions = X[col] == 1
            ratios.append(col + '\t' + str(mean(y['Priority'][conditions])) + '\t\t\t\t(' + str(mean(y['Priority'][conditions]) * sum(X[col][conditions])) + '/' + str(sum(X[col][conditions])) + ')')
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


if __name__ == '__main__':
    fb = Facebook()
    dfs = fb.load_all_fbs()
    fb_users = fb.merge_dfs(dfs)
    fb_users = fb.combine_all_features(fb_users) 
    fb_users = fb.binarize_gender(fb_users)
    fb_users = fb.only_users_with_X(fb_users, ['gender', 'name'])
    locales = fb.binarize_locales(fb_users)
    X = fb.create_X(fb_users, locales)
    y = fb.create_y(fb_users)
    model, score = fb.build_model(X, y)
    important_cols = fb.find_important_features(model)
    print score, 'accuracy score'
