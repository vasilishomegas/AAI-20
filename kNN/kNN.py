from scipy.spatial import distance
from import_data import *


def find_kNN(new_point, k):
	# calculate distances
	distances = []
	for point in normalised_data:
		distances.append(distance.euclidean(new_point, point))
	# link distance to season label
	season_dist = []
	for x in range(0, len(labels)):
		season_dist.append((distances[x], labels[x]))
	season_dist.sort()
	# count the occurrence of each season
	seasons = {'winter': 0, 'lente': 0, 'zomer': 0, 'herfst': 0}
	for result in season_dist[:k]:
		seasons[result[1]] += 1
	# group seasons by occurrence
	seasoncount = {}
	for k, v in seasons.items():
		if v in seasoncount:
			seasoncount[v].append(k)
		else:
			seasoncount[v] = [k]
	# if there is one season that appears most commonly in the result
	if len(seasoncount[max(seasoncount.keys())]) == 1:
		return seasoncount[max(seasoncount.keys())][0]
	# if more than one season is the most frequently appearing
	else:
		for day in season_dist:
			if day[1] in seasoncount[max(seasoncount.keys())]:
				return day[1]


def validate_kNN(max_k):
	# final accuracy per k
	k_accuracy = []
	for k in range(1, max_k+1):
		# number of correct answers so far
		accuracy = 0
		# go through all validation data
		for day in range(0, len(normalised_validation_data)):
			season = find_kNN(normalised_validation_data[day], k)
			if season == validation_labels[day]:
				accuracy += 1
		k_accuracy.append(accuracy/len(normalised_validation_data))
	for x in range(0, len(k_accuracy)):
		print(x+1, ":", k_accuracy[x]*100, "%")


validate_kNN(20)
for date in normalised_test_data:
	print(find_kNN(date, 19))