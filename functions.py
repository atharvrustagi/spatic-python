import numpy as np
from functools import lru_cache

# A function to check for similarity between two names
def is_similar_str(word1: str, word2: str, max_edits = 5) -> bool:
    len1, len2 = len(word1), len(word2)
    # recursive function to check for similarity by comparing each character
    # returns the number of edits that need to be made in order to make word1[i:] == word2[j:]
    @lru_cache
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

# Function to find the distance between two points
def distance(point1: tuple, point2: tuple) -> float:
    # acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1))*ER
    ER = 6371000 # Earth's radius in meters
    lat1, lon1 = np.deg2rad(point1)
    lat2, lon2 = np.deg2rad(point2)
    return np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1)) * ER
