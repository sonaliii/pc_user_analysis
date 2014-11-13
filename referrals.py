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


class Referrals(object):
    def __init__(self):
        self.user_df = pd.read_csv('../data/u.csv', sep=',', error_bad_lines=False, dtype={'UserID': 'object', 'SourceUserID': 'object', 'Email': 'object', 'FacebookID': 'object', 'CircleIDs': 'object'})
        self.weeks = pd.read_csv('../data/week_counts.csv', sep=',', error_bad_lines=False, header=None)
        self.weeks.columns = ['user_id', 'weeks']
        self.users = None

    def join_referrals_weeks(self):
        referrals = self.user_df[['UserID', 'SourceUserID']].dropna()
        weeks = self.weeks
        referral_weeks = referrals.merge(weeks, how='inner', left_on='UserID', right_on='user_id')
        referral_weeks = referral_weeks.groupby('SourceUserID')
        summed = referral_weeks.aggregate({'weeks': np.sum})
        summed = summed.reset_index()
        summed.columns = ['UserID', 'Referrals']
        return summed

    def priority_retention(self):
        retention = self.weeks
        retention['Priority'] = retention['weeks'].apply(lambda x: True if x >= 52 else False)
        users = self.user_df.merge(retention, how='outer', left_on='UserID', right_on='user_id')
        #Replacing NAs with 0 referrals and False priority
        users['weeks'] = users['weeks'].fillna(0)
        users['Priority'] = users['Priority'].fillna(False)

        self.users = users
        return users

    def priority_by_weeks(self, referral_weeks_df):
        referrers = referral_weeks_df
        #Setting priority users to those who have referred at least 10 other users
        referrers['Priority'] = referrers['Referrals'].apply(lambda x: True if x >= 10 else False)
        
        #Trying various limits for priority number of referrals
        for i in range(10):
            referrers['Priority' + str(i + 1)] = referrers['Referrals'].apply(lambda x: True if x >= i + 1 else False)
    
        #Merging all users with those who have referred others
        users = self.user_df.merge(referrers, how='outer', left_on='UserID', right_on='UserID')
        
        #Replacing NAs with 0 referrals and False priority
        users['Referrals'] = users['Referrals'].fillna(0)
        users['Priority'] = users['Priority'].fillna(False)
        
        self.users = users
        return users


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
    # referrers = r.join_referrals_weeks()
    retention = r.priority_retention()
    # users = r.priority_by_weeks(retention)
    r.count_circles(retention)
