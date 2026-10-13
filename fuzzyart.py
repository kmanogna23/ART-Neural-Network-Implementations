
import numpy as np

X = np.array([
    [0.9, 0.8],
    [0.8, 0.9],
    [0.1, 0.2],
    [0.2, 0.1]
])

vigilance = 0.7
alpha = 0.001
clusters = []

for pattern in X:
    assigned = False

    for i, prototype in enumerate(clusters):
        intersection = np.minimum(pattern, prototype).sum()
        choice = intersection / (alpha + prototype.sum())

        match = intersection / (pattern.sum() + 1e-10)

        if match >= vigilance:
            clusters[i] = np.minimum(pattern, prototype)
            print(pattern, "belongs to Cluster", i + 1)
            assigned = True
            break

    if not assigned:
        clusters.append(pattern.copy())
        print(pattern, "belongs to Cluster", len(clusters))
