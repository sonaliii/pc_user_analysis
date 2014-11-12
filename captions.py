import pandas as pd
import base64

# #Importing to Pandas dataframe
# df = pd.read_csv('../data/captions.txt', sep=',', error_bad_lines=False)

# #Grouping comments by user ID
# grouped = df.groupby('user_id')
# joined = grouped.aggregate({'comment_text': ' '.join})
# joined = joined.reset_index()
# joined['user_id'] = joined['user_id'].apply(lambda x: x.split(',', 2))

# #Saving to CSV
# joined.to_csv('../data/captions.csv', encoding='utf-8')

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

joined.to_csv('../data/captions.csv', encoding='utf-8')

