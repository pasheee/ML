import numpy as np


# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """

    des1_sq = np.sum(des1**2, axis=1, keepdims=True)  # (N1, 1)
    des2_sq = np.sum(des2**2, axis=1)                 # (N2,)
    dot_product = np.dot(des1, des2.T)                # (N1, N2)
    distance_sq = des1_sq + des2_sq - 2 * dot_product  # (N1, N2)


    matches1 = np.argmin(distance_sq, axis=1)  
    matches2 = np.argmin(distance_sq, axis=0)  


    valid_mask = (matches2 < des1.shape[0]) & (matches1[matches2] == np.arange(des2.shape[0]))
    t_indices = np.where(valid_mask)[0]
    q_indices = matches2[t_indices]   


    distances = np.sqrt(distance_sq[q_indices, t_indices])


    matches = [
        DummyMatch(int(q), int(t), float(d))
        for q, t, d in zip(q_indices, t_indices, distances)
    ]

    matches.sort(key=lambda x: (x.distance, x.queryIdx, x.trainIdx))

    return matches

