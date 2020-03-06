import numpy as np

data = np.genfromtxt("../data/dataset1.csv", delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates = np.genfromtxt("../data/dataset1.csv", delimiter=";", usecols=[0])
labels = []
for label in dates:
	if label < 20000301:
		labels.append("winter")
	elif 20000301 <= label < 20000601:
		labels.append("lente")
	elif 20000601 <= label < 20000901:
		labels.append("zomer")
	elif 20000901 <= label < 20001201:
		labels.append("herfst")
	else:  # from 01-12 to end of year
		labels.append("winter")

fg_min = data[0][0]
fg_max = data[0][0]
tg_min = data[0][1]
tg_max = data[0][1]
tn_min = data[0][2]
tn_max = data[0][2]
tx_min = data[0][3]
tx_max = data[0][3]
sq_min = data[0][4]
sq_max = data[0][4]
dr_min = data[0][5]
dr_max = data[0][5]
rh_min = data[0][6]
rh_max = data[0][6]

for date in data:
	if date[0] < fg_min:
		fg_min = date[0]
	elif date[0] > fg_max:
		fg_max = date[0]
	if date[1] < tg_min:
		tg_min = date[1]
	elif date[1] > tg_max:
		tg_max = date[1]
	if date[2] < tn_min:
		tn_min = date[2]
	elif date[2] > tn_max:
		tn_max = date[2]
	if date[3] < tx_min:
		tx_min = date[3]
	elif date[3] > tx_max:
		tx_max = date[3]
	if date[4] < sq_min:
		sq_min = date[4]
	elif date[4] > sq_max:
		sq_max = date[4]
	if date[5] < dr_min:
		dr_min = date[5]
	elif date[5] > dr_max:
		dr_max = date[5]
	if date[6] < rh_min:
		rh_min = date[6]
	elif date[6] > rh_max:
		rh_max = date[6]

fg_range = fg_max - fg_min
tg_range = tg_max - tg_min
tn_range = tn_max - tn_min
tx_range = tx_max - tx_min
sq_range = sq_max - sq_min
dr_range = dr_max - dr_min
rh_range = rh_max - rh_min


def normalise(date):
	return [
		(date[0] - fg_min) / fg_range,
		(date[1] - tg_min) / tg_range,
		(date[2] - tn_min) / tn_range,
		(date[3] - tx_min) / tx_range,
		(date[4] - sq_min) / sq_range,
		(date[5] - dr_min) / dr_range,
		(date[6] - rh_min) / rh_range]


normalised_data = []
for date in data:
	# normalised_date = normalise(date)
	normalised_data.append(normalise(date))

# for n in range(0, len(dates)):
# 	print(dates[n], ":", labels[n])

validation_data = np.genfromtxt("../data/validation1.csv", delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validation_dates = np.genfromtxt("../data/validation1.csv", delimiter=";", usecols=[0], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validation_labels = []
for label in validation_dates:
	if label < 20010301:
		validation_labels.append("winter")
	elif 20010301 <= label < 20010601:
		validation_labels.append("lente")
	elif 20010601 <= label < 20010901:
		validation_labels.append("zomer")
	elif 20010901 <= label < 20011201:
		validation_labels.append("herfst")
	else:  # from 01-12 to end of year
		validation_labels.append("winter")

normalised_validation_data = []
for date in validation_data:
	normalised_validation_data.append(normalise(date))

test_data = np.genfromtxt("../data/days.csv", delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
normalised_test_data = []
for date in test_data:
	normalised_test_data.append(normalise(date))


neural_network_data = np.genfromtxt("../data/bezdekIris.data", delimiter=",", usecols=[0, 1, 2, 3])
neural_network_classification = np.genfromtxt("../data/bezdekIris.data", delimiter=",", usecols=[4], converters={4: lambda s: 0 if s == "Iris-setosa" else 1 if s == "Iris-versicolor" else 2 if s == "Iris-virginica" else 3})
