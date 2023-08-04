import sys
import os

home_path = '.'


def work():
	f = open(home_path + "/data/one_step_tmp/fea_list.txt")
	fea_list = []
	for line in f:
		value = line.strip().split("\t")
		if len(value) > 0:
			fea_list = value
	f.close()

	d = {}
	f = open(home_path + "/data/all_theta_dict.txt")
	for line in f:
		line = line.strip()
		value = line.split("\t")
		if len(value) != 2:
			continue
		d[value[0]] = value[1]
	f.close()

	n = len(fea_list)
	theta = []
	for i in range(0, n):
		fea = fea_list[i]
		if fea in d:
			theta.append(d[fea])
		else:
			theta.append("0.0")
	f = open(home_path + "/data/one_step_tmp/theta.txt", "w")
	f.write("\t".join(theta) + "\n")
	f.close()
	print("write theta.txt finish")


if __name__ == "__main__":
	work()


