import pandas as pd
import numpy as np
from time import perf_counter as pf
from functools import lru_cache

EARTH_RADIUS = 6371009          # Earth's radius in meters
DISTANCE_THRESHOLD = 200.0      # maximum distance in meters

def is_similar_str(word1: str, word2: str, max_edits = 5) -> bool:
    """A function to check for similarity between two names"""
    len1, len2 = len(word1), len(word2)

    # recursive function to check for similarity by comparing each character
    # returns the number of edits that need to be made in order to make word1[i:] == word2[j:]
    @lru_cache  # to store already computed results (dynamic programming)
    def num_edits(i: int, j: int) -> int:
        # base cases
        if i == len1:
            # there are no more characters to compare in word1
            # so, in order to make strings similar, delete all remaining characters of word2
            return len2-j
        if j == len2:
            # similar to the case for i == len1
            return len1-i
        
        if word1[i] == word2[j]:
            # if the characters match, just move forward without any edits
            return num_edits(i+1, j+1)
        # characters do not match, so editing is required
        edits = 10000
        # case 1 -> delete i'th character from word1, which is the same as insertion in word2
        edits = min(edits, 1+num_edits(i+1, j))
        # case 2 -> delete j'th character from word2, which is the same as insertion in word1
        edits = min(edits, 1+num_edits(i, j+1))
        # case 3 -> replace any of the two characters
        edits = min(edits, 1+num_edits(i+1, j+1))
        return edits
    
    return num_edits(0, 0) <= max_edits

def distance(point1: tuple, point2: tuple) -> float:
    """
    Same as geopy.distance.great_circle
    But supports numpy arrays as well.
    Function to find the distance between two points in meters
    """
    lat1, lon1 = np.deg2rad(point1)
    lat2, lon2 = np.deg2rad(point2)
    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)
    sin_delta_lon, cos_delta_lon = np.sin(lon2-lon1), np.cos(lon2-lon1)
    d = np.arctan2(np.sqrt((cos_lat2 * sin_delta_lon) ** 2 +
                    (cos_lat1 * sin_lat2 -
                    sin_lat1 * cos_lat2 * cos_delta_lon) ** 2),
                sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lon)
    return EARTH_RADIUS * d

if __name__ == '__main__':
    # starting timer for benchmarking purposes
    t = pf()
    data = pd.read_csv('assignment_data.csv')
    N = data.shape[0]
    # boolean array to store whether similarity is there or not in an entry
    similar = [0] * N

    """
    # Brute force method, comparing each point against all other points to check for similarity
    # This method's runtime is very high
    # Approximating the runtime
    # It takes 139.16 seconds for i = 0:50
    # total iterations for i = 0:50 -> 589425
    # Time taken for each iteration -> 139.16 / 589425 = 0.000236 seconds
    # Total iterations for i = 0:11814 -> ((11814-1) + (11814-2) + .. (11814-11814)) -> 11814*(11814-1)/2 = 69779391
    # Time required for 69779391 iterations -> 0.000236 * 69779391 = 16478 seconds or 4.57 hours
    from geopy.distance import great_circle
    for i in range(50):
        print(i)
        point1 = (data['latitude'][i], data['longitude'][i])
        for j in range(i+1, N):
            point2 = (data['latitude'][j], data['longitude'][j])
            if great_circle(point1, point2) <= DISTANCE_THRESHOLD and is_similar_str(data['name'][i], data['name'][j]):
                similar[i] = similar[j] = 1
    """

    # An optimization, where we calculate the distances to all other points at once
    # for every point or location, compare it against all other points
    # Runtime -> ~ 21 seconds
    for i in range(N):
        point1 = (data['latitude'][i], data['longitude'][i])
        # get the distance of current point to all points after current point
        distances = distance(point1, (data['latitude'][i+1:], data['longitude'][i+1:]))
        for j in range(i+1, N):
            if distances[j-i-1] <= DISTANCE_THRESHOLD and is_similar_str(data['name'][i], data['name'][j]):
                similar[i] = similar[j] = 1

    data['is_similar'] = similar
    data.to_csv('output.csv', index=False)

    print(f'Done in {pf()-t} seconds')
    
