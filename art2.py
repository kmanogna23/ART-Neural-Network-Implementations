
import numpy as np

X = np.array([
    [0.9, 0.8],
    [0.8, 0.9],
    [0.1, 0.2],
    [0.2, 0.1]
])

vigilance = 0.5
clusters = []

for pattern in X:
    assigned = False

    for i, prototype in enumerate(clusters):
        similarity = np.dot(pattern, prototype) / (
            np.linalg.norm(pattern) * np.linalg.norm(prototype)
        )

        if similarity >= vigilance:
            clusters[i] = (prototype + pattern) / 2
            print(pattern, "belongs to Cluster", i + 1)
            assigned = True
            break

    if not assigned:
        clusters.append(pattern.copy())
        print(pattern, "belongs to Cluster", len(clusters))
