import numpy as np

# Two simple feature vectors
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

# Difference between the vectors
difference = x - y

# Euclidean distance
distance = np.linalg.norm(difference)

print("Vector x:", x)
print("Vector y:", y)
print("Difference:", difference)
print("Euclidean distance:", distance)