import pandas as pd
from time import perf_counter as pf
from functions import is_similar_str, distance

t = pf()

DIST_THRESHOLD = 200.0    # in meters
data = pd.read_csv('assignment_data.csv')
N = data.shape[0]
# boolean array to store whether similarity is there or not
similar = [0] * N
# for every entry, compare it against all other entries
for i in range(N):
    point1 = (data['latitude'][i], data['longitude'][i])
    # get the distance of current point to all points after current point
    distances = distance(point1, (data['latitude'][i+1:], data['longitude'][i+1:]))
    for j in range(i+1, N):
        if distances[j-i-1] <= DIST_THRESHOLD and is_similar_str(data['name'][i], data['name'][j]):
            similar[i] = similar[j] = 1
print()
data['is_similar'] = similar
# print(data)
data.to_csv('output.csv', index=False)

print(f'Done in {pf()-t} seconds')
