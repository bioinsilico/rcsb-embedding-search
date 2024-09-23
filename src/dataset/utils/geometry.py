import math

import numpy as np


def normal_vector(p1, p2, p3):
    # Calculate vectors from the points
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)

    # Calculate the cross product to get the normal vector
    normal = np.cross(v1, v2)
    return normal


def angle_between_planes(points1, points2):
    # Calculate the normal vectors for each plane
    normal1 = normal_vector(*points1)
    normal2 = normal_vector(*points2)

    # Normalize the normal vectors
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = normal2 / np.linalg.norm(normal2)

    # Calculate the dot product of the normal vectors
    dot_product = np.dot(normal1, normal2)

    # Ensure the dot product is within the valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians
    return np.arccos(dot_product)


def angle_between_vectors(v1, v2):
    # Convert to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Ensure the cosine value is within the valid range for arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians
    return np.arccos(cos_angle)


def angle_between_points(p1, p2, p3):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2

    return angle_between_vectors(v1, v2)


def angle_between_four_points(p1, p2, q1, q2):
    # Convert to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    q1 = np.array(q1)
    q2 = np.array(q2)

    return angle_between_vectors(p2 - p1, q2 - q1)


def distance_between_points(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)


def exp_distance(d):
    return math.exp(-(d - 3.6) ** 2 / 12)


if __name__ == "__main__":
    # Example usage:
    points_plane1 = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    points_plane2 = [(0, 0, 0), (0, 0, 1), (0, 1, 1)]

    angle_rad = angle_between_planes(points_plane1, points_plane2)
    print(f"Angle between planes: {angle_rad} radians ({np.degrees(angle_rad)} degrees)")

    # Example usage:
    vector1 = [1, 0, 0]
    vector2 = [0, 1, 0]

    angle_rad = angle_between_vectors(vector1, vector2)
    print(f"Angle between vectors: {angle_rad} radians ({np.degrees(angle_rad)} degrees)")
