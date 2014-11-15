import pandas as pd
'''
Imports caption text data, groups by User ID, and saves to CSV
'''
lines = []
with open('../data/captions.txt') as f:
  lines = f.readlines()

split_lines = []
for line in lines:
  split_line = line.split(',', 2)
  split_lines.append(split_line)

df = pd.DataFrame(split_lines, columns=['user_id', 'circle_id', 'caption'])
df = df.drop('circle_id', axis=1)
grouped = df.groupby('user_id')
joined = grouped.aggregate({'caption': ' '.join})
joined = joined.reset_index()

joined.to_csv('../data/captions.csv')

