import pandas as pd
import sqlite3
'''
Imports SQLite database of comment text, groups by User ID, and saves to CSV
'''

#Connecting to SQLite Database and importing to Pandas dataframe
con = sqlite3.connect('comments.sqlite')
df = pd.read_sql("SELECT user_id, comment_text FROM comments", con)

#Grouping comments by user ID
grouped = df.groupby('user_id')
joined = grouped.aggregate({'comment_text': ' '.join})
joined = joined.reset_index()

#Saving to CSV
joined.to_csv('../data/comments.csv', encoding='utf-8')
