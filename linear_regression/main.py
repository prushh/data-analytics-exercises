import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay


def df_split(df_input, df_target, test_size: float = 0.33, random_state: int = 20) -> tuple:
    return train_test_split(
        df_input,
        df_target,
        test_size=test_size,
        random_state=random_state
    )


def compute_metrics(target_test, quality_target_pred: np.ndarray) -> tuple:
    return mean_squared_error(target_test, quality_target_pred), \
           r2_score(target_test, quality_target_pred)


def linear_regression(df_input, df_target) -> tuple:
    input_train, input_test, target_train, target_test = df_split(df_input, df_target)
    reg = LinearRegression()
    reg.fit(input_train, target_train)
    quality_target_pred = reg.predict(input_test)

    mse, r_square = compute_metrics(target_test, quality_target_pred)
    print(f'Mean Square Error: {mse}')
    print(f'R-square: {r_square}')
    return input_test, target_test, quality_target_pred


# 1
FILE = 'dataset/quality.csv'

df = pd.read_csv(FILE)
print(f'-- Dataset {FILE} uploaded --')
print(f'Row: {df.shape[0]}, Column: {df.shape[1]}')

# 2
df_clean = df.dropna()
print('-- Size after clean --')
print(f'Row: {df_clean.shape[0]}')

# 3
print('-- Linear regression with X = TEMP, Y = PM2.5 --')
df_x = df_clean['TEMP'].values.reshape((-1, 1))
df_y = df_clean['PM2.5']
x_test, y_test, quality_y_pred = linear_regression(df_x, df_y)

# 4
plt.scatter(x_test, y_test, color='green')
plt.xlabel('TEMP')
plt.ylabel('PM2.5')
plt.plot(x_test, quality_y_pred, color='red', linewidth=3)
plt.show()

# 5
print('-- Linear regression with all X and Y = PM2.5 --')
df_all_x = df_clean.drop(columns=['No', 'PM2.5', 'wd', 'station'])
df_y = df_clean['PM2.5']

df_wd = pd.get_dummies(df_clean['wd'])
df_station = pd.get_dummies(df_clean['station'])

df_all_x = pd.concat([df_all_x, df_wd, df_station], axis=1)

_, _, _ = linear_regression(df_all_x, df_y)

# 6
print('-- Logistic regression with all X and Y = station --')
df_x = df_clean.drop(columns=['No', 'wd', 'station'])
df_y = df_clean['station']

df_wd = pd.get_dummies(df_clean['wd'])

df_x = pd.concat([df_x, df_wd], axis=1)

x_train, x_test, y_train, y_test = df_split(df_x, df_y)
# TODO: solve ConvergenceWarning
# clf = LogisticRegression(random_state=20, multi_class='multinomial', max_iter=500)
clf = LogisticRegression(random_state=20, multi_class='multinomial', verbose=1, max_iter=1500)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print('Avg accuracy: ', clf.score(x_test, y_test))

# Plot the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
display.plot()
plt.show()
