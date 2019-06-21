import csv

csvdir = "../realsense-data/csv/"

with open(csvdir+"_Depth_100.csv") as f:
    reader = csv.reader(f)
    l = [row for row in reader]

print(l[100][100])
