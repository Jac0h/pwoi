import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from pyransac3d import Plane

# --- Wczytywanie xyz ---
def load_xyz(filename):
    points = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if row:
                points.append(list(map(float, row)))
    return np.array(points)

# --- Własny RANSAC dla płaszczyzny ---
def fit_plane(points):
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, p1)
    return normal, d

def point_to_plane_distance(point, normal, d):
    return np.abs(np.dot(normal, point) + d) / np.linalg.norm(normal)

def ransac_plane(points, num_iterations=1000, distance_threshold=0.01):
    best_inliers = []
    best_plane = (None, None)

    for _ in range(num_iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        normal, d = fit_plane(sample)

        distances = np.abs(np.dot(points, normal) + d)
        inliers = points[distances < distance_threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers

# --- Analiza płaszczyzny ---
def analyze_plane(normal, inliers):
    distances = [point_to_plane_distance(p, normal, 0) for p in inliers]
    mean_distance = np.mean(distances)
    print(f"Średnia odległość punktów od płaszczyzny: {mean_distance:.6f}")

    if mean_distance < 0.02:
        print("To jest płaszczyzna.")
        normal = np.abs(normal)
        if normal[2] > 0.8:
            print("Płaszczyzna pozioma (wzdłuż Z).")
        elif normal[0] > 0.8 or normal[1] > 0.8:
            print("Płaszczyzna pionowa (wzdłuż X lub Y).")
        else:
            print("Płaszczyzna skośna.")
    else:
        print("To NIE jest płaszczyzna.")

# --- Podział chmur na klastry ---
def split_cloud_kmeans(points, k=3):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(points)
    clusters = []
    for i in range(k):
        clusters.append(points[labels == i])
    return clusters, labels

def split_cloud_dbscan(points, eps=1.5, min_samples=10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    clusters = []
    for label in np.unique(labels):
        if label != -1:
            clusters.append(points[labels == label])
    return clusters, labels

# --- Wizualizacja klastrów ---
def visualize_clusters(points, labels, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Szum (noise)
            color = 'k'
            label_name = 'Noise'
        else:
            label_name = f'Cluster {label}'
        ax.scatter(points[labels == label, 0], points[labels == label, 1], points[labels == label, 2],
                   s=5, color=color, label=label_name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

# --- Główna funkcja ---
def main():
    # Wczytaj punkty
    points = load_xyz("all_clouds.xyz")

    # Podziel na chmury
    clusters_kmeans, labels_kmeans = split_cloud_kmeans(points, k=3)
    clusters_dbscan, labels_dbscan = split_cloud_dbscan(points, eps=1.5, min_samples=10)

    # --- Wizualizacja ---
    visualize_clusters(points, labels_kmeans, "KMeans Clustering (k=3)")
    visualize_clusters(points, labels_dbscan, "DBSCAN Clustering")

    # --- Własny RANSAC ---
    print("\n== Własny RANSAC + KMeans ==")
    for i, cluster in enumerate(clusters_kmeans):
        print(f"\n--- Chmura {i+1} ---")
        plane, inliers = ransac_plane(cluster)
        normal, d = plane
        print(f"Wektor normalny: {normal}, d={d}")
        analyze_plane(normal, inliers)

    # --- pyransac3d ---
    print("\n== Pyransac3d + DBSCAN ==")
    for i, cluster in enumerate(clusters_dbscan):
        print(f"\n--- Chmura {i+1} ---")
        plane_model = Plane()
        (model, inliers) = plane_model.fit(cluster, thresh=0.01)

        if len(inliers) < 0.5 * len(cluster):
            print("Za mało punktów dopasowało się do płaszczyzny — prawdopodobnie to NIE jest płaszczyzna.")
            continue

        a, b, c, d = model
        normal = np.array([a, b, c])

        print(f"Wektor normalny: {normal}, d={d}")
        analyze_plane(normal, cluster[inliers])

if __name__ == "__main__":
    main()
