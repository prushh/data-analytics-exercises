import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def data_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state
    )


def svm(x: np.ndarray, y: np.ndarray, kernel: str = 'linear', regularization: int = 1, degree: int = 0,
        gamma: str = ''):
    # Try with others random_state
    x_train, x_test, y_train, y_test = data_split(x, y)

    model = None
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=regularization)
    elif kernel == 'poly':
        model = SVC(kernel=kernel, C=regularization, degree=degree)
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, C=regularization, gamma=gamma)

    model.fit(x_train, y_train)
    print(f'Accuracy [test, {kernel}]: {model.score(x_test, y_test):.2f}')


def main():
    data = datasets.load_wine()
    x_data = data['data']
    y_target = data['target']

    svm(x_data, y_target)
    svm(x_data, y_target, kernel='poly', degree=3)
    svm(x_data, y_target, kernel='rbf', gamma='scale')


if __name__ == '__main__':
    main()
