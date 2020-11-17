from copy import deepcopy
import scipy.io.wavfile
import sys
import numpy as np
np.set_printoptions(suppress=True)

max_iterations = 30

# Function: should_stop
# Returns True or False if k-means is done.
# run a maximum number of iterations or the centroids stop changing.


def should_stop(prev_centroids, cur_centroids, num_of_iteration):
    if num_of_iteration >= max_iterations:
        return True
    # convergence
    return (cur_centroids == prev_centroids).all()


# Function: k_Means
# K-Means is an algorithm that takes training_set_set and centroids
# and define clusters of data in the dataset which are similar to one another
# centroids - array of points - the centroids
# x - training_set
# fs - rate of wav file
def k_means(centroids, x, fs):
    output_file = open(f"output2.txt", "w")
    num_of_clusters = centroids.shape
    num_of_training_set = x.shape
    num_of_iterations = 0
    prev_centroids = np.zeros(num_of_clusters)
    cur_centroids = np.zeros(num_of_clusters)
    clusters_size = np.zeros(num_of_clusters[0])
    new_value = []

    while not should_stop(prev_centroids, centroids, num_of_iterations):
        loss = 0
        prev_centroids = deepcopy(centroids)
        # Calculate Distances Between each example in training_set From All Other centroids
        for i in range(num_of_training_set[0]):
            min_dist_value = np.linalg.norm(x[i] - prev_centroids[0])
            closest_centroid = 0
            for j in range(num_of_clusters[0]):
                dist = np.linalg.norm(x[i] - prev_centroids[j])
                # In case when 2 centroids are evenly close to a certain point,
                # the one with the lower index ”wins
                if dist < min_dist_value:
                    min_dist_value = dist
                    closest_centroid = j
            loss += np.power(min_dist_value, 2)
            # Assign cur x[i] to closest_centroid
            cur_centroids[closest_centroid] += x[i]
            clusters_size[closest_centroid] += 1
            new_value.append(prev_centroids[closest_centroid])
        # Update the centroids to average of all the point in is cluster
        for i in range(num_of_clusters[0]):
            if cur_centroids[i].shape[0] > 0:
                centroids[i] = np.round(cur_centroids[i] / clusters_size[i])
        loss /= num_of_training_set[0]
        print(loss)
        cur_centroids = np.zeros(num_of_clusters)
        clusters_size = np.zeros(num_of_clusters[0])
        output_file.write(f"[iter {num_of_iterations}]:{','.join([str(i) for i in centroids])}\n")
        num_of_iterations += 1
        scipy.io.wavfile.write("compressed.wav", fs, np.array(new_value, dtype=np.int16))

# Function: random_centroids
# random k centroids over training_set for example i


def random_centroids(training_set, k, i):
    training_set_size = training_set.shape[0]
    centroids = [[0 for x in range(2)] for y in range(k)]
    for j in range(k):
        r = np.random.randint(0, training_set_size - 1)
        centroids[j] = training_set[r]
    np.savetxt(f"Example{i}.txt", centroids, delimiter=' ')


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)
    k_means(centroids, x, fs)
"""
k_values = [2, 4, 8, 16]
    i = 1
    for k in k_values:
        random_centroids(x, k, i)
        i += 1

"""




if __name__ == '__main__':
    main()
