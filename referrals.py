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
import matplotlib
import seaborn

# %pylab inline

#Loading app user data
# u = pd.read_csv('../u.csv', sep=',', error_bad_lines=False, dtype={'UserID': 'object', 'SourceUserID': 'object', 'Email': 'object', 'FacebookID': 'object', 'CircleIDs': 'object'})

class Referrals(object):
    def __init__(self):
        self.user_df = pd.read_csv('../data/u.csv', sep=',', error_bad_lines=False, dtype={'UserID': 'object', 'SourceUserID': 'object', 'Email': 'object', 'FacebookID': 'object', 'CircleIDs': 'object'})
        self.users = None

    def count_referrals(self):
        users = self.user_df
        referrals = users[['UserID', 'SourceUserID']]
        referrals = referrals.dropna()
        
        #Counting the number of occurrences of a referral by each SourceUserID
        referrers = referrals.groupby('SourceUserID').count()
        
        #Removing repeated column and sorting by number of referrals
        referrers = referrers.drop('SourceUserID', axis = 1)
        referrers = referrers.sort(columns='UserID', ascending=False)
        
        #Resetting index to make SourceUserID a column again
        referrers = referrers.reset_index()
        
        #Renaming column with count of referrals
        referrers.columns = ['SourceUserID', 'Referrals']

        return referrers

    def identify_priority_users(self, referrers_df):
        referrers = referrers_df
        #Setting priority users to those who have referred at least 10 other users
        referrers['Priority'] = referrers['Referrals'].apply(lambda x: True if x >= 1 else False)
        
        #Trying various limits for priority number of referrals
        for i in range(10):
            referrers['Priority' + str(i + 1)] = referrers['Referrals'].apply(lambda x: True if x >= i + 1 else False)
    
        #Merging all users with those who have referred others
        users = self.user_df.merge(referrers, how='outer', left_on='UserID', right_on='SourceUserID')
        users = users.drop('SourceUserID_x', axis = 1)
        users = users.drop('SourceUserID_y', axis = 1)
        
        #Replacing NAs with 0 referrals and False priority
        users['Referrals'] = users['Referrals'].fillna(0)
        users['Priority'] = users['Priority'].fillna(False)
        
        self.users = users
        return users

    def count_circles(self, priority_users):
        users = self.users
        
        #Counting the number of circles each user belongs to
        users['has_circles'] = users['CircleIDs'].apply(lambda x: not isinstance(x, float))
        users['Circles'] = users['CircleIDs'][users['has_circles'] == True].apply(lambda x: len(x.split(';')))
        self.users = users
        return users

    def plot_referrals_distribution(referrers):
        #Looking at the distribution of the number of referrals per user
        hist(referrers['Referrals'], bins = 100)
        plt.title('Number of referrals per user')
        plt.xlabel('Number referred')
        plt.ylabel('Number of users')
    
if __name__=='__main__':
    r = Referrals()
    referrers = r.count_referrals()
    users = r.identify_priority_users(referrers)
    r.count_circles(users)
