import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FILE = 'dataset/wine.data'
TARGET = 'Class'


def get_df(filename: str, target: str, fields: list = None) -> tuple:
    if fields is None:
        df = pd.read_csv(filename)
    else:
        fields.append(target)
        df = pd.read_csv(filename, usecols=fields)

    df_x = df.loc[:, df.columns != target]
    df_y = pd.Series(df[target])

    return df_x, df_y


def df_split(df_x: pd.DataFrame, df_y: pd.Series, test_size: float = 0.33, random_state: int = 100) -> tuple:
    return train_test_split(
        df_x,
        df_y,
        test_size=test_size,
        random_state=random_state
    )


def decision_tree(df_x: pd.DataFrame, df_y: pd.Series, max_depth: int = 6, criterion: str = 'gini',
                  ccp_alpha: float = 0.0035, log: bool = False, plot: bool = False) -> tuple:
    x_train, x_test, y_train, y_test = df_split(df_x, df_y)

    dtc = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        ccp_alpha=ccp_alpha
    )

    dtc.fit(x_train, y_train)
    if plot:
        tree.plot_tree(dtc, filled=True)
        plt.show()

    dtc.predict(x_test)
    test_accuracy = dtc.score(x_test, y_test)
    train_accuracy = dtc.score(x_train, y_train)

    if log:
        text_representation = tree.export_text(dtc)
        print(text_representation)
        print(f'Avg accuracy [{criterion}]: {test_accuracy}')

    return train_accuracy, test_accuracy


def random_forest(df_x: pd.DataFrame, df_y: pd.Series, max_depth: int = 4, n_estimators: int = 20, n_variables: int = 3,
                  log: bool = False, plot: bool = False) -> tuple:
    x_train, x_test, y_train, y_test = df_split(df_x, df_y)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    rf.fit(x_train, np.ravel(y_train))
    rf.predict(x_test)
    rank_var = pd.Series(
        rf.feature_importances_,
        index=df_x.columns
    ).sort_values(ascending=False)
    test_accuracy = rf.score(x_test, y_test)
    train_accuracy = rf.score(x_test, y_test)

    if log:
        print(f'Avg accuracy [random_forest]: {test_accuracy}')
        print(rank_var)

    if plot:
        sns.barplot(x=rank_var, y=rank_var.index)
        plt.xlabel('Variable Importance Score')
        plt.ylabel('Variables')
        plt.show()

    return list(rank_var.index[:n_variables]), train_accuracy, test_accuracy


def gradient_boosting(df_x: pd.DataFrame, df_y: pd.Series, n_estimators: int = 20, learning_rate: float = .1,
                      log: bool = False) -> tuple:
    x_train, x_test, y_train, y_test = df_split(df_x, df_y)

    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    gb.fit(x_train, y_train)
    gb.predict(x_test)
    test_accuracy = gb.score(x_test, y_test)
    train_accuracy = gb.score(x_train, y_train)

    if log:
        print(f'Avg accuracy [gradient_boosting]: {test_accuracy}')

    return train_accuracy, test_accuracy


def repeat_classification(parameters: np.ndarray, learning_rates=None, n_repetitions: int = 30,
                          technique: str = 'decision_tree', plot: bool = False) -> None:
    train_accuracies, test_accuracies = [], []
    for parameter in parameters:
        avg_alpha_train, avg_alpha_test = [], []
        for _ in range(n_repetitions):
            train_accuracy, test_accuracy = [], []
            if technique == 'decision_tree':
                train_accuracy, test_accuracy = decision_tree(*get_df(FILE, TARGET), ccp_alpha=parameter)
            elif technique == 'random_forest':
                _, train_accuracy, test_accuracy = random_forest(*get_df(FILE, TARGET), n_estimators=parameter)
            elif technique == 'gradient_boosting' and learning_rates is not None:
                train_accuracy, test_accuracy = gradient_boosting(*get_df(FILE, TARGET), n_estimators=parameter,
                                                                  learning_rate=learning_rates)
            avg_alpha_train.append(train_accuracy)
            avg_alpha_test.append(test_accuracy)
        train_accuracies.append(np.mean(avg_alpha_train))
        test_accuracies.append(np.mean(avg_alpha_test))

    if plot:
        x_axis = np.arange(len(parameters))

        plt.bar(x_axis - 0.2, train_accuracies, 0.4, label='Train')
        plt.bar(x_axis + 0.2, test_accuracies, 0.4, label='Test')

        nice_technique = technique.capitalize().replace('_', ' ')
        title = f'{nice_technique} - {n_repetitions} repetitions'
        x_label = 'Alpha' if technique == 'decision_tree' else 'Num. estimators'
        y_label = 'Accuracy value'

        plt.xticks(x_axis, parameters)
        plt.xlabel(x_label)
        plt.title(title)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()


def main() -> int:
    # 1
    decision_tree(*get_df(FILE, TARGET))
    decision_tree(*get_df(FILE, TARGET), criterion='entropy')
    # 2
    mr_variables, _, _ = random_forest(*get_df(FILE, TARGET), plot=True)
    random_forest(*get_df(FILE, TARGET, fields=mr_variables), plot=True)
    # 3
    alphas = np.array([0, .005, .01, .02, .05, .1])
    repeat_classification(alphas)
    # 4
    n_estimators = np.array([1, 10, 20, 50, 100, 200])
    repeat_classification(n_estimators, technique='random_forest')
    # 5
    # TODO: missing repetitions
    gradient_boosting(*get_df(FILE, TARGET), log=True)

    return 0


if __name__ == '__main__':
    main()
