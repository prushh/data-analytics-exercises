import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/estate.csv', sep=';')

# 1
sns.boxplot(y=df['Distance'], whis=1.5)
plt.show()

# 2
sns.scatterplot(x=df['Age'], y=df['Price'])
plt.show()

sns.scatterplot(x=df['Distance'], y=df['Price'])
plt.show()

# 3
fig, axs = plt.subplots(2, 3)
sns.kdeplot(df['Age'], ax=axs[0, 0])
sns.kdeplot(df['Distance'], ax=axs[0, 1])
sns.scatterplot(x=df['Distance'], y=df['Price'], ax=axs[0, 2])
sns.boxplot(y=df['Age'], whis=1.5, ax=axs[1, 0])
sns.boxplot(y=df['Distance'], whis=1.5, ax=axs[1, 1])
sns.scatterplot(x=df['Age'], y=df['Price'], ax=axs[1, 2])

plt.show()
