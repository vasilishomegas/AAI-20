from scipy.spatial import distance
from operator import add
import functools
from import_data import *
import random


class Cluster:
    def __init__(self, centroid, cluster_id):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.points = []
        self.old_points = None
        self.changed = True

    def move_centroid(self,points):
        self.centroid = list(map((lambda p: p / len(self.points)),list(map((lambda i: sum(list(map((lambda x: x[i]),points)))),list(range(0,len(points[0])))))))

    def recalculate_centroid(self):
        self.move_centroid(self.points) if len(self.points) else (self.move_centroid(self.old_points) if len(self.old_points) else print("Error: old did not contain any points"))

    def check_points_change(self):
        self.changed = self.points != self.old_points

    def clear_points(self):
        self.check_points_change()
        self.old_points = self.points
        self.points = []

    def calculate_avg_dist_to_centroid(self):
        dist_sum = 0
        for point in self.old_points:
            dist_sum += distance.euclidean(point, self.centroid)
        return dist_sum / len(self.old_points)

    def __repr__(self):
        string = "Centroid: "
        string += "".join(str(self.centroid))
        string += "\nId: " + str(self.cluster_id) + "\nPoints:\n"
        for point in self.points:
            string += "".join(str(point)) + "\n"
        return string


def generate_centroid():
    # pick random values within the limits of the dataset
    return normalised_validation_data[random.randint(0, len(normalised_validation_data))]


def get_centroids(k):
    centroids = []
    for i in range(0, k):
        centroid = generate_centroid()
        if centroid not in centroids:  # so we don't pick the same centroid twice
            centroids.append(centroid)
        else:
            i -= 1
    return centroids


def check_all_clusters_changed(clusters):
    for cluster in clusters:
        if cluster.changed:
            return True
    return False


def kMeans(cls):
    while check_all_clusters_changed(clusters):
        for point in normalised_validation_data:  # go through all points
            distances = []
            for cl in cls:  # calculate distance to each centroid
                # add distance from point to current centroid, and the id of the cluster of that centroid
                distances.append([distance.euclidean(point, cl.centroid), cl.cluster_id])
            distances.sort()  # sort distances so that the shortest will come first
            for cl in cls:  # check which cluster corresponds to the cluster_id with the shortest distance
                if cl.cluster_id == distances[0][1]:
                    cl.points.append(point)  # add point to list of points of cluster
                    break
        # so now we have clustered every point around a centroid
        # let's recalculate each centroid based on its assigned points
        for cluster in clusters:
            print(cluster)
            cluster.recalculate_centroid()
            print(cluster)
            cluster.clear_points()
            print(cluster)


k = 4
cts = get_centroids(k)
clusters = []
for i in range(0, k):
    clusters.append(Cluster(cts[i], i))

kMeans(clusters)
