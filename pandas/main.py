import pandas as pd

# 1
df = pd.DataFrame({
    'Anno': [2018, 2014, 2010, 2006, 2002],
    'Sede': ['Russia', 'Brasile', 'Sudafrica', 'Germania', 'Corea del Sud'],
    'Vincitore': ['Francia', 'Germania', 'Spagna', 'Italia', 'Brasile'],
    'Numero_Goal': [169, 171, 145, 147, 161],
    'Pubblico': [47371, 53592, 49670, 52401, 42268]
})

df.to_csv('dataset/world_cup.csv', sep='|', index=False)
df_world = pd.read_csv('dataset/world_cup.csv', sep='|')

# 2
df_world['Spettacolo_OK'] = df_world['Numero_Goal'] > 150

# 3
winners = df_world.head(3)['Vincitore']

# 4
df_world['Numero_Goal'] = df_world['Numero_Goal'] / 64

# 5
location = df_world.loc[df_world['Numero_Goal'].argmin(), 'Sede']

# 6
df_euro = pd.read_csv('dataset/european_cup.csv', sep='|')
teams = pd.merge(df_world, df_euro, how='inner', on='Vincitore')['Vincitore'].drop_duplicates()

# 7
df_all = pd.concat([df_world, df_euro], ignore_index=True)

df_all['Tipologia'] = 'NaN'
df_all['Tipologia'] = df_all['Anno'].apply(lambda year: 'EUROPEI' if year % 4 == 0 else 'MONDIALI')

# 8
max_public = df_all.groupby('Tipologia')['Pubblico'].max()
min_public = df_all.groupby('Tipologia')['Pubblico'].min()
result = max_public - min_public
