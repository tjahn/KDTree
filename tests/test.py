# %%

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.joinpath("build/PyKDTree")))

import PyKDTree

# %

rng = np.random.default_rng(7)
points = rng.random((1000, 2))

lineNNQueries = rng.random((0, 2, 2)).astype(np.float32)
linekNNQueries = rng.random((0, 2, 2)).astype(np.float32)
lineRadiusQueries = rng.random((1, 2, 2)).astype(np.float32)

pointNNQueries = rng.random((0, 2)).astype(np.float32)
pointkNNQueries = rng.random((0, 2)).astype(np.float32)
pointRadiusQueries = rng.random((0, 2)).astype(np.float32)


tree = PyKDTree.KDTree2d(points)


plt.plot(points[:, 0], points[:, 1], "b.", ms=2)
plt.axis("equal")
plt.grid()


for point in pointNNQueries:
    res = tree.point_nn_search(point)
    result = points[res[0]]
    plt.plot(result[0], result[1], "go")

    plt.plot(point[0], point[1], "ro")

for point in pointkNNQueries:
    res = tree.point_knn_search(point, 50)
    for r in res:
        result = points[r[0]]
        plt.plot(result[0], result[1], "go", alpha=0.5)

    plt.plot(point[0], point[1], "ro")
    plt.gca().add_patch(
        plt.Circle(point, np.max(np.array(res)[:, 1]) ** 0.5, color="r", alpha=0.3)
    )

for point in pointRadiusQueries:
    radius = 0.2

    res = tree.point_radius_search(point, radius)
    for r in res:
        result = points[r[0]]
        plt.plot(result[0], result[1], "go", alpha=0.5)

    plt.plot(point[0], point[1], "ro")
    plt.gca().add_patch(plt.Circle(point, radius, color="r", alpha=0.3))

for a, b in lineNNQueries:
    res = tree.line_nn_search(a, b)
    result = points[res[0]]
    plt.plot(result[0], result[1], "go", alpha=0.5)

    plt.plot(a[0], a[1], "ro")
    plt.plot(b[0], b[1], "ro")
    plt.plot((a[0], b[0]), (a[1], b[1]), "r-")

for a, b in linekNNQueries:
    ress = tree.line_knn_search(a, b, 100)
    for res in ress:
        result = points[res[0]]
        plt.plot(result[0], result[1], "go", alpha=0.5)

    radius = np.max(np.array(ress)[:, 1]) ** 0.5
    plt.plot(a[0], a[1], "ro")
    plt.plot(b[0], b[1], "ro")
    plt.gca().add_patch(plt.Circle(a, radius, color="r", alpha=0.3))
    plt.gca().add_patch(plt.Circle(b, radius, color="r", alpha=0.3))
    plt.plot((a[0], b[0]), (a[1], b[1]), "r-")
    dx, dy = b[1] - a[1], b[0] - a[0]
    dx, dy = (
        -dx * radius / (dx**2 + dy**2) ** 0.5,
        dy * radius / (dx**2 + dy**2) ** 0.5,
    )
    plt.plot((a[0] + dx, b[0] + dx), (a[1] + dy, b[1] + dy), "r-")
    plt.plot((a[0] - dx, b[0] - dx), (a[1] - dy, b[1] - dy), "r-")

for a, b in lineRadiusQueries:
    radius = 0.1
    ress = tree.line_radius_search(a, b, radius)
    for res in ress:
        result = points[res[0]]
        plt.plot(result[0], result[1], "go", alpha=0.5)

    plt.plot(a[0], a[1], "ro")
    plt.plot(b[0], b[1], "ro")
    plt.gca().add_patch(plt.Circle(a, radius, color="r", alpha=0.3))
    plt.gca().add_patch(plt.Circle(b, radius, color="r", alpha=0.3))
    plt.plot((a[0], b[0]), (a[1], b[1]), "r-")
    dx, dy = b[1] - a[1], b[0] - a[0]
    dx, dy = (
        -dx * radius / (dx**2 + dy**2) ** 0.5,
        dy * radius / (dx**2 + dy**2) ** 0.5,
    )
    plt.plot((a[0] + dx, b[0] + dx), (a[1] + dy, b[1] + dy), "r-")
    plt.plot((a[0] - dx, b[0] - dx), (a[1] - dy, b[1] - dy), "r-")

# %%
