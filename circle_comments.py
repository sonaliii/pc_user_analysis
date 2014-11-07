import pandas as pd
import sqlite3

con = sqlite3.connect('../../../../comments.sqlite')
df = pd.read_sql("SELECT circle_id, comment_text FROM comments", con)

#Grouping comments by circle ID
grouped = df.groupby('circle_id')
joined = grouped.aggregate({'comment_text': ' '.join})
joined = joined.reset_index()

#Saving to CSV
joined.to_csv('circle_comments.csv', encoding='utf-8')