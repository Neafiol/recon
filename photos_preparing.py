import json
import os
import pickle
import warnings

import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

photo_url = "https://admin5500.s3.eu-central-1.amazonaws.com"
with open("photos.pk", "rb") as f:
    photos = pickle.load(f)

done = os.listdir("data")
photos_name = [p["name"] for p in photos]
data = []


def pfile(f):
    for line in f.readlines():
        if line[-1] == "\n":
            line = line[:-1]
        dat = json.loads(line)
        data.append(dat)


with open("learning.jsonlines", "r") as f:
    pfile(f)

with open("input.jsonlines", "r") as f:
    pfile(f)

print(data)

def load(url):
    name = url.split("/")[-1]
    if name in done:
        return
    try:
        with open(f"data/{name}", "wb") as f:
            r = requests.get(photo_url + url)
            f.write(r.content)
            done.append(name)
    except Exception as e:
        print(f"Error {e}: {url}")


dnames = [d["photo"].split("/")[-1] for d in data]
print("Start loading photos")
for d in tqdm(data):
    load(d["photo"])

print("Start preparing photos:", len(done) - len(photos_name))
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tf_pose

for i, d in tqdm(enumerate(data)):
    if d['photo'] in photos_name:
        continue
    try:
        coco_style = tf_pose.infer(f"/home/petr/Documents/Projects/Recognition/data/{d['photo']}")
    except Exception as e:
        print(e, f"data/{d['photo']}")
        continue
    photos.append({
        "name": d['photo'] ,
        "points": coco_style,
        "shape": None
    })
    photos_name.append(d['photo'])
    if i % 150 == 0:
        with open("photos.pk", "wb") as f:
            pickle.dump(photos, f)

with open("photos.pk", "wb") as f:
    pickle.dump(photos, f)

print("Complete", len(data), "total:", len(photos))
