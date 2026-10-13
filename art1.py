
import numpy as np

# Binary input patterns
X = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

vigilance = 0.7
clusters = []

for pattern in X:
    assigned = False

    for i, prototype in enumerate(clusters):
        intersection = np.sum(pattern * prototype)
        union = np.sum(pattern)

        if union != 0 and intersection / union >= vigilance:
            clusters[i] = np.minimum(prototype, pattern)
            print(pattern, "belongs to Cluster", i + 1)
            assigned = True
            break

    if not assigned:
        clusters.append(pattern.copy())
        print(pattern, "belongs to Cluster", len(clusters))
