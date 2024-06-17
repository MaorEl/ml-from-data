import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    X_flat = X.reshape(-1, X.shape[-1])
    random_indices = np.random.choice(X_flat.shape[0], k, replace=False)
    centroids = X_flat[random_indices]

    return centroids.astype(float)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    X_flat = X.reshape(-1, X.shape[-1])

    distances = []
    for centroid in centroids:
        distance = np.linalg.norm(X_flat - centroid, ord=p, axis=1)
        distances.append(distance)

    distances = np.array(distances)

    return distances


def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    X_flat = X.reshape(-1, X.shape[-1])
    classes = np.zeros(X_flat.shape[0], dtype=int)
    centroids, classes, converged_at = kmeans_bl(X_flat, centroids, classes, k, max_iter, p)

    return centroids, classes


def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None

    X_flat = X.reshape(-1, X.shape[-1])

    # Step 1 - Choose a centroid uniformly at random among the data points
    size = len(X_flat)
    centroids = [X_flat[np.random.choice(size)]]

    # Step 4 - Repeat Steps 2 and 3 until k centroids are chosen
    for _ in range(1, k):
        # Step 2 - Compute the distance between each data point and the nearest centroid
        linalg = np.linalg
        centorids_distances = [linalg.norm(X_flat - c, ord=p, axis=1) for c in centroids]
        dist = np.min(centorids_distances, axis=0)

        # Step 3 - Choose one new data point at random as a new centroid
        dist = dist / dist.sum()
        centroid = X_flat[np.random.choice(size, p=dist)]
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Step 5 - Proceed using standard k-means clustering
    centroids, classes, converged_at = kmeans_bl(X_flat, centroids, classes, k, max_iter, p)

    return centroids, classes


def kmeans_bl(X_flat, centroids, classes, k, max_iter, p):
    converge_at = None
    for _ in range(max_iter):
        distances = lp_distance(X_flat, centroids, p)
        classes = np.argmin(distances, axis=0)
        new_centroids = np.array([X_flat[classes == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            converge_at = _
            # print(f'Converged after {converge_at} iterations.')
            break

        centroids = new_centroids

    return centroids, classes, converge_at


def kmeans_pp_performance(X, k, p ,max_iter=100):
    classes = None
    X_flat = X.reshape(-1, X.shape[-1])
    size = len(X_flat)
    centroids = [X_flat[np.random.choice(size)]]
    for _ in range(1, k):
        linalg = np.linalg
        centorids_distances = [linalg.norm(X_flat - c, ord=p, axis=1) for c in centroids]
        dist = np.min(centorids_distances, axis=0)
        dist = dist / dist.sum()
        centroid = X_flat[np.random.choice(size, p=dist)]
        centroids.append(centroid)

    centroids = np.array(centroids)
    centroids, classes, converged_at = kmeans_bl(X_flat, centroids, classes, k, max_iter, p)

    return centroids, classes, converged_at


def kmeans_performance(X, k, p ,max_iter=100):
    centroids = get_random_centroids(X, k)
    X_flat = X.reshape(-1, X.shape[-1])
    classes = np.zeros(X_flat.shape[0], dtype=int)
    centroids, classes, converged_at = kmeans_bl(X_flat, centroids, classes, k, max_iter, p)

    return centroids, classes, converged_at


def calculate_sum_squared_distances(image, centroids, classes):
    ssd = 0
    for i in range(image.shape[0]):
        ssd += np.sum((image[i] - centroids[classes[i]])**2)

    return ssd