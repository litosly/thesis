import os 
import json 
import tqdm 

path = "yelp_academic_dataset_review.json"
# path = "test.json"
with open(path, 'r') as f:
	data = f.readlines()
	data = list(map(json.loads, data))

total_len = len(data)
num = 10
chunk_len = total_len // num

folder = "data"
if not os.path.exists(folder):
	os.makedirs(folder)
for i in range(num):
	name = os.path.join(folder, "%d.txt" % i) 
	with open(name, 'a') as f:
		for line in tqdm.tqdm(data[i*chunk_len :  (i+1)*chunk_len]):
			data_s = json.dumps(line)+"\n"
			f.write(data_s)
	print("fininshed file: ", i)

name = os.path.join(folder, "%d.txt" % num) 
with open(name, 'a') as f:
	for line in tqdm.tqdm(data[num*chunk_len :  total_len]):
		data_s = json.dumps(line)+"\n"
		f.write(data_s)
print("fininshed file: ", i)
