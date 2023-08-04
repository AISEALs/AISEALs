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

	theta = []
	f = open(home_path + "/data/one_step_tmp/theta.txt")
	for line in f:
		value = line.strip().split("\t")
		if len(value) > 0:
			theta = value
	f.close()

	n = len(fea_list)
	m = len(theta)
	if (n != m):
		return
	for i in range(0, n):
		fea = fea_list[i]
		value = theta[i]
		d[fea] = value

	f = open(home_path + "/data/one_step_tmp/all_theta_dict.txt", "w")
	f2 = open(home_path + "/data/all_theta_record.txt", "a")
	for item in d:
		f.write(item + "\t" + d[item] + "\n")
		f2.write(item + ":" + d[item] + "\t")
	f2.write("\n")
	f.close()
	f2.close()
	


if __name__ == "__main__":
	work()


