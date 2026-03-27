from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    """
    Evaluate clustering performance using multiple metrics

    Parameters:
    X → input data
    labels → cluster labels

    Returns:
    Dictionary of scores
    """

    results = {}

    # Silhouette Score (range: -1 to 1, higher is better)
    results["Silhouette Score"] = silhouette_score(X, labels)

    # Davies-Bouldin Index (lower is better)
    results["Davies-Bouldin Index"] = davies_bouldin_score(X, labels)

    # Calinski-Harabasz Score (higher is better)
    results["Calinski-Harabasz Score"] = calinski_harabasz_score(X, labels)

    return results