import pandas as pd
import sqlite3

#Connecting to SQLite Database and importing to Pandas dataframe
con = sqlite3.connect('comments.sqlite')
df = pd.read_sql("SELECT user_id, comment_text FROM comments", con)

#Grouping comments by user ID
grouped = df.groupby('user_id')
joined = grouped.aggregate({'comment_text': ' '.join})
joined = joined.reset_index()

#Saving to CSV
joined.to_csv('comments.csv', encoding='utf-8')