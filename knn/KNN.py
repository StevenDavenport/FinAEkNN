import numpy as np
from collections import Counter

class KNN:
    """
    K-Nearest Neighbors classifier.
    This algoorithm will be used on financial data.
    """

    def __init__(self, k=8):
        self.k = k
        max_bars_back = 2000
        feature_count = 5

    def fit(self, X, y):
        """
        Fit the model using the training data.
        Parameters:
        - X: Training data features, a NumPy array of shape (n_samples, n_features)
        - y: Training data labels, a NumPy array of shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        Parameters:
        - X: Test data, a NumPy array of shape (n_query, n_features)
        Returns:
        - predictions: A list of predicted class labels.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predict the class label for a single sample.
        Parameters:
        - x: A single sample, a NumPy array of shape (n_features,)
        Returns:
        - predicted_label: The predicted class label for the sample.
        """
        # Compute the distances between x and all examples in the training set
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Get the k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        predicted_label = most_common[0][0]
        return predicted_label

    def _lorentzian_distance(self, x, y):
        """
        Compute the Lorentzian distance between two samples.
        Parameters:
        - x: A single sample, a NumPy array of shape (n_features,)
        - y: A single sample, a NumPy array of shape (n_features,)
        Returns:
        - distance: The Lorentzian distance between the two samples.
        """
        distance = np.sum(np.log(1 + np.abs(x - y)))
        return distance

    def _euclidean_distance(self, x, y):
        """
        Compute the Euclidean distance between two samples.
        Parameters:
        - x: A single sample, a NumPy array of shape (n_features,)
        - y: A single sample, a NumPy array of shape (n_features,)
        Returns:
        - distance: The Euclidean distance between the two samples.
        """
        distance = np.sqrt(np.sum((x - y) ** 2))
        return distance

    def _manhattan_distance(self, x, y):
        """
        Compute the Manhattan distance between two samples.
        Parameters:
        - x: A single sample, a NumPy array of shape (n_features,)
        - y: A single sample, a NumPy array of shape (n_features,)
        Returns:
        - distance: The Manhattan distance between the two samples.
        """
        distance = np.sum(np.abs(x - y))
        return distance

    def _cosine_similarity(self, x, y):
        """
        Compute the Cosine similarity between two samples.
        Parameters:
        - x: A single sample, a NumPy array of shape (n_features,)
        - y: A single sample, a NumPy array of shape (n_features,)
        Returns:
        - similarity: The Cosine similarity between the two samples.
        """
        similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return similarity

    def _jaccard_similarity(self, x, y):
        """
        Compute the Jaccard similarity between two samples.
        Parameters:
        - x: A single sample, a NumPy array of shape (n_features,)
        - y: A single sample, a NumPy array of shape (n_features,)
        Returns:
        - similarity: The Jaccard similarity between the two samples.
        """
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        similarity = np.sum(intersection) / np.sum(union)
        return similarity





























