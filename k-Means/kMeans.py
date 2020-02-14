from scipy.spatial import distance
import numpy as np
from operator import add, concat
import functools
import math
from import_data import *
import random


class Cluster:
    def __init__(self, centroid, cluster_id):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.points = []
        self.old_points = None
        self.changed = True

    def calc_pos_centroid(self,points):
        #moves the centroid to the average of the points provided as parameter.
        self.centroid = list(map((lambda p: p / len(points)),list(map((lambda i: sum(list(map((lambda x: x[i]),points)))),list(range(0,len(points[0])))))))

    def recalculate_centroid(self):
        #checks if the centroid has points, if no points are currently available use the old points.
        self.calc_pos_centroid(self.points) if len(self.points) else print("Error: old did not contain any points")

    def check_points_change(self):
        #checks if the points have moved.
        self.changed = self.points != self.old_points if self.old_points != [] else True

    def clear_points(self):
        self.check_points_change()
        self.old_points = self.points
        self.points = []

    def calculate_avg_dist_to_centroid(self):
        # calculates the avg distance to the centroid of the containing points, if a centroid does not have any point it returns infinite!
        return (sum(list(map((lambda point: distance.euclidean(point,self.centroid)),self.old_points)))) / len(self.old_points) if len(self.old_points) else math.inf

    def __repr__(self):
        string = "Centroid: "
        string += "".join(str(self.centroid))
        string += "\nId: " + str(self.cluster_id) + "\nPoints:\n"
        for point in self.points:
            string += "".join(str(point)) + "\n"
        return string

def efficiency(cls):
    return sum(list(map((lambda cl: cl.calculate_avg_dist_to_centroid()),cls))) / len(cls)

def generate_centroid():
    # pick random values within the limits of the dataset
    return normalised_data[random.randint(0, len(normalised_data)-1)]


def get_centroids(k):
    centroids = []
    while len(centroids) != k:
        centroid = generate_centroid()
        if centroid not in centroids:  # so we don't pick the same centroid twice
            centroids.append(centroid)
    return centroids


def any_cluster_changed(cls):
    #check if any of the clusters has changed, if so return true
    return True in list(map((lambda cl: cl.changed),cls))


def kMeans(cls):
    while any_cluster_changed(cls):
        for point in normalised_data:  # go through all points
            # add distance from point to current centroid, and the id of the cluster of that centroid
            distances = sorted(list(map((lambda cl: [distance.euclidean(point,cl.centroid),cl.cluster_id]),cls)))
            for cl in cls:  # check which cluster corresponds to the cluster_id with the shortest distance
                if cl.cluster_id == distances[0][1]:
                    cl.points.append(point)  # add point to list of points of cluster
                    break
        # so now we have clustered every point around a centroid
        # let's recalculate each centroid based on its assigned points
        for cluster in cls:
            cluster.recalculate_centroid()
            cluster.clear_points()
    return efficiency(cls)

def kmeans_mult(cls,num):
    return (sorted(list(map(kMeans,[cls]*num))))[0] if num > 0 else math.inf

def optimal_k(recalculate_num=1):
    k = 1
    efficiency_results = [kmeans_mult(init(k),recalculate_num)]
    while True:
        k += 1
        efficiency_results.append(kmeans_mult(init(k),recalculate_num))
        temp = np.diff(efficiency_results, 2)
        if temp.size > 0 and temp[-1] <= 0:
            k -= 1
            print(efficiency_results)
            print(temp)
            break
    return k

def init(k):
    return list(map((lambda xs: Cluster(xs[1],xs[0])),list(enumerate(get_centroids(k)))))


print(optimal_k(30))
