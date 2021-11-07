import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_decision_boundaries(x, y, model_class, **model_params):
    """
    shorturl.at/bkG47
    Function to plot the decision boundaries of a classification model.
    This uses just the first two columns of the data for fitting
    the model as we need to find the predicted value for every point in
    scatter plot.
    Arguments:
            x: Feature data as a NumPy-type array.
            y: Label data as a NumPy-type array.
            model_class: A Scikit-learn ML estimator class
            e.g. GaussianNB (imported from sklearn.naive_bayes) or
            Logistic Regression (imported from sklearn.linear_model)
            **model_params: Model parameters to be passed on to the ML estimator

    Typical code example:
            plt.figure()
            plt.title("KNN decision boundary with neighbros: 5",fontsize=16)
            plot_decision_boundaries(X_train,y_train,KNeighborsClassifier,n_neighbors=5)
            plt.show()
    """
    try:
        x = np.array(x)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    # reduced_data = x[:, 2:4]
    reduced_data = x
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01  # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = x[:, 2].min() - 1, x[:, 2].max() + 1
    y_min, y_max = x[:, 3].min() - 1, x[:, 3].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, z, alpha=0.4)
    plt.scatter(x[:, 2], x[:, 3], c=y, alpha=0.8)
    plt.xlabel("Petal Length", fontsize=15)
    plt.ylabel("Petal Width", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return plt


# 1
df = pd.read_csv('dataset/fire.csv')
print('-- Load dataset --')
print(f'Row: {df.shape[0]}')
print(f'Column: {df.shape[1]}')

# 2
df_clean = df.dropna()
print('-- Size after clean --')
print(f'Row: {df_clean.shape[0]}')
print(f'Column: {df_clean.shape[1]}')

# 3
df_x = df_clean.drop(columns=['Classes'])
df_y = df_clean['Classes'].str.strip()

le = LabelEncoder()
le.fit(df_y)
df_y = le.transform(df_y)

print(f'Co-variance values: {np.cov(df_x, rowvar=False)}')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=20)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
prediction = gnb.predict(x_test)
print(f'Avg accuracy: {gnb.score(x_test, y_test)}')

cm = confusion_matrix(y_test, prediction, labels=gnb.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
display.plot()
plt.show()

# 4
# plot_decision_boundaries(x, y, GaussianNB)
