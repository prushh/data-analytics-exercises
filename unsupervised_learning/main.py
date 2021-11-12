import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_df(path: str = 'dataset/wine.data', y_col: str = 'Class') -> tuple:
    data = pd.read_csv(path)
    data_y = data[y_col]
    data.drop(columns=y_col, inplace=True)
    return data, data_y


def k_means(data: pd.DataFrame, n_clusters: int = 3, init: str = 'k-means++', max_iter: int = 300, n_init: int = 10,
            random_state: int = 0, label: str = 'no_scaling', log: bool = False) -> tuple:
    model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init,
                   random_state=random_state)
    model.fit(data)
    if log:
        print(f'Inertia [{label}]: {model.inertia_}')
    return model.inertia_, model.labels_


def gaussian_mixture(data: pd.DataFrame, n_components: int = 3, log: bool = False) -> np.ndarray:
    model = GaussianMixture(n_components=n_components).fit(data)
    if log:
        distribution = model.predict_proba(data)
        print(distribution)
    labels = model.predict(df)
    return labels


def hierarchical_clustering(data: pd.DataFrame, log: bool = False) -> AgglomerativeClustering:
    model = AgglomerativeClustering(compute_full_tree=True, affinity='euclidean', linkage='ward')
    model.fit(data)
    if log:
        print(model.labels_)
    return model


def preprocessing(data: pd.DataFrame, method: str) -> pd.DataFrame:
    scaler = None
    if method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'standardization':
        scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return scaled_data


def elbow(data: pd.DataFrame) -> None:
    wcv = []
    rng = range(1, 15)
    for k in rng:
        inertia, _ = k_means(data, n_clusters=k)
        wcv.append(inertia)
    plt.plot(rng, wcv)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Within-Cluster Variation')
    plt.show()
    print('Applied elbow method')


def plot_scatter(x: pd.Series, y: pd.Series, labels: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].scatter(x=x, y=y, c=labels['y'], s=40, cmap='viridis')
    ax[0, 1].scatter(x=x, y=y, c=labels['gmm'], s=40, cmap='viridis')
    ax[1, 0].scatter(x=x, y=y, c=labels['k_means'], s=40, cmap='viridis')
    ax[1, 1].scatter(x=x, y=y, c=labels['k_means'], s=40, cmap='viridis')

    plt.show()


def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


# 1
df, y_labels = get_df()
k_means(df)
# 2
scaler_label = 'standardization'
df = preprocessing(df, scaler_label)
k_means(df, label=scaler_label, log=True)

scaler_label = 'min-max'
df = preprocessing(df, scaler_label)
_, k_means_labels = k_means(df, label=scaler_label, log=True)
# 3
elbow(df)
# 4
gmm_labels = gaussian_mixture(df)
all_labels = pd.DataFrame({
    'y': y_labels,
    'gmm': gmm_labels,
    'k_means': k_means_labels
})
plot_scatter(df['Alcohol'], df['Hue'], all_labels)
# 5
clustering = hierarchical_clustering(df)
plot_dendrogram(clustering, truncate_mode='level', p=5)
