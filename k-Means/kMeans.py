from scipy.spatial import distance
import math
from statistics import median
import matplotlib.pyplot as plt
from import_data import *
import random


# Point class for easier storage and access of the position and label of points
class Point:
    def __init__(self, pos, label):
        self.pos = pos
        self.label = label


points = list(map(lambda x: Point(x[0], x[1]), zip(normalised_data, labels)))  # Global data, used throughout kMeans


# Cluster class for kMeans
class Cluster:
    def __init__(self, centroid, cluster_id):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.points = []
        self.old_centroid = None
        self.changed = True

    def calc_pos_centroid(self):
        # moves the centroid to the average of the points provided as parameter.
        self.centroid = list(map((lambda p: p / len(self.points)), list(
            map((lambda i: sum(list(map((lambda x: x.pos[i]), self.points)))),
                list(range(0, len(self.points[0].pos)))))))

    def check_centroid_changed(self):
        # checks if the points have moved.
        self.changed = self.centroid != self.old_centroid if self.old_centroid is not None else True

    def recalculate_centroid(self):
        # checks if the centroid has points, if no points are currently available use the old points.
        self.old_centroid = self.centroid
        self.calc_pos_centroid() if len(self.points) else print("Error: no points for the centroid to be recalculated.")
        self.check_centroid_changed()

    def clear_points(self):
        self.points = []

    def get_cluster_label(self):
        possible_labels = {}
        for point in self.points:
            if point.label in possible_labels:
                possible_labels[point.label] += 1
            else:
                possible_labels[point.label] = 1
        # returns the label that occurs the most in the points in this cluster.
        return max(possible_labels, key=possible_labels.get) if possible_labels is not {} else "Error: No label for this cluster! Probable cause: No points inside this centroids."

    def total_dist_from_points_to_centroid(self):
        # returns the sum of all the distances the the centroid.
        return sum(list(map((lambda point: distance.euclidean(point.pos, self.centroid)), self.points)))

    def calculate_avg_dist_to_centroid(self):
        # calculates the avg distance to the centroid of the containing points, if a centroid does not have any point it returns infinite!
        return self.total_dist_from_points_to_centroid() / len(self.points) if len(self.points) else math.inf

    def __repr__(self):
        string = "Centroid: "
        string += "".join(str(self.centroid))
        string += "\nId: " + str(self.cluster_id) + "\nPoints:\n"
        for point in self.points:
            string += "".join(str(point.pos)) + "\n"
        return string


def efficiency(cls):
    # returns the sum of the distances from all points to its corresponding clusters
    return sum(list(map((lambda cl: cl.total_dist_from_points_to_centroid()), cls)))


def generate_centroid():
    # pick random values within the limits of the dataset
    return normalised_data[random.randint(0, len(normalised_data) - 1)]


def get_centroids(k):
    # Returns list with length k filled with positions of the centroids.
    centroids = []
    while len(centroids) != k:
        centroid = generate_centroid()
        if centroid not in centroids:  # so we don't pick the same centroid twice
            centroids.append(centroid)
    return centroids


def any_cluster_changed(cls):
    # check if any of the clusters has changed, if so return true
    return True in list(map((lambda cl: cl.changed), cls))


def kMeans(cls):
    while any_cluster_changed(cls):
        for cluster in cls:
            cluster.clear_points() #clear the clusters.
        for point in points:  # go through all points
            # add distance from point to current centroid, and the id of the cluster of that centroid
            distances = sorted(list(map((lambda cl: [distance.euclidean(point.pos, cl.centroid), cl.cluster_id]), cls)))
            for cl in cls:  # check which cluster corresponds to the cluster_id with the shortest distance
                if cl.cluster_id == distances[0][1]:
                    cl.points.append(point)  # add point to list of points of cluster
                    break
        # so now we have clustered every point around a centroid
        # let's recalculate each centroid based on its assigned points
        for cluster in cls:
            cluster.recalculate_centroid() #recalculate the position of the cluster.
    return cls


def kmeans_mult(cls, num):
    # Runs kmeans and recalculates num amount of times using cls as its starting centroids returns the best efficiency one.
    return (sorted(list(map(lambda x: efficiency(kMeans(x)), [cls] * num))))[0] if num > 0 else math.inf


def optimal_k(recalculate=1,redo=1):
    # Gets two values
    # Recalculate : the amount of times the individual k are calculated. (best performing k is selected)
    # Redo : the amount of times the kMeans algorithm is performed. (median is chosen, then converted to int. Any floats are floored.)
    def inner(recalculate_num,cur_iter):
        k = 1 # Starting k
        efficiency_results = [kmeans_mult(init(k), recalculate_num)] #efficiency list that contains the best functioning kMeans for each k
        while True:
            k += 1
            efficiency_results.append(kmeans_mult(init(k), recalculate_num))
            snd_diff = np.diff(efficiency_results, 2) # Calculates the second derivative of the efficiency results
            if snd_diff.size > 0 and snd_diff[-1] <= 0: #when the second derivative is below 0 the best functioning k is the previous one.
                k -= 1
                print("[" + str(cur_iter+1) + "/" + str(redo) + "]".format(1,1))
                break
        return k, efficiency_results, snd_diff

    print()

    results = list(map(lambda i: inner(recalculate,i), list(range(0,redo)))) #get all results from the inner function and puts them in a list
    plt.title("All efficiency results")
    plt.xlabel("K")
    plt.ylabel("Efficiency (lower = better)")
    for result in results:
        plt.plot(range(1,len(result[1])+1), result[1])  # plots every efficiency result
    plt.show()  # show all results

    plt.title("All efficiency results")
    plt.xlabel("K")
    plt.ylabel("2nd derivative of efficiency")
    max_length = 0
    for result in results:
        if len(result[2]) > max_length:
            max_length = len(result[2])
        plt.plot(range(2, len(result[2])+2), result[2])
    # max_length += 2
    plt.plot(range(2, max_length+2), [0]*max_length)
    plt.show()

    print()
    return int(median(map(lambda x: x[0], results)))  # gets the median of the inner function.


def init(k):
    # initial function to generate k amount of clusters.
    return list(map((lambda xs: Cluster(xs[1], xs[0])), list(enumerate(get_centroids(k)))))


def get_label_closest_cluster(cls,point):
    # gets the label of the point using the provided centroids/clusters. The label of the closest cluster is returned.
    return cls[sorted(list(map((lambda cl: [distance.euclidean(point, cl.centroid), cl.cluster_id]), cls)))[0][1]].get_cluster_label()


def check_clustering(cls):
    right = 0
    for g,v in zip(list(map(lambda p: get_label_closest_cluster(cls,p), normalised_validation_data)),validation_labels):
        right += g == v
    return right / len(validation_labels) * 100


num_cls = 4
print(list(map((lambda cl: cl.get_cluster_label()), kMeans(init(num_cls))))) # prints all cluster labels.
print("K: " + str(optimal_k(3,10))) # finds the optimal k for this dataset and returns it.

# print(check_clustering(kMeans(init(optimal_k(3,10)))))
