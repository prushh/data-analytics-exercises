import numpy as np
import pandas as pd


def f(group):
    return group[group.y == 'yes']['y'].count()


# 1 - Dataset upload
df = pd.read_csv('dataset/bank_dataset.csv', sep=';')
print('-- Size before clean --')
print(f'Row: {df.shape[0]}')
print(f'Column: {df.shape[1]}', end='\n\n')

# 2 - Clean NaN values
df_clean = df.dropna()
print('-- Size after clean --')
print(f'Row: {df_clean.shape[0]}')
print(f'Column: {df_clean.shape[1]}', end='\n\n')

# 3 - Clean duplicates values
columns = df_clean.columns
df_last = df_clean.drop_duplicates(columns)
print('-- Size after duplicates --')
print(f'Row: {df_last.shape[0]}')
print(f'Column: {df_last.shape[1]}', end='\n\n')

# 4 - Mean, max and min for age column
mean_age = np.mean(df_last['age'])
max_age = np.max(df_last['age'])
min_age = np.min(df_last['age'])
print(f'Mean age: {mean_age}')
print(f'Max age: {max_age}')
print(f'Min age: {min_age}', end='\n\n')

# 5
count_user = df_last.loc[df_last.education == 'university.degree']['age'].count()
print('-- Count user with university.degree --')
print(f'User: {count_user}', end='\n\n')

# 6
df_job_bank = df_last.groupby('job').apply(f)
print('-- Group by job and count for each user who has opened a bank account  --')
print(df_job_bank, end='\n\n')

# 7
df_one_hot = pd.concat([pd.get_dummies(df_last.education), df_last], axis=1)
print('-- OneHotEncoding on education column --')
print(f'Column: {df_one_hot.shape[1]}')
