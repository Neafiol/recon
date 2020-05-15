import os
import pickle
import sys
import warnings
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

photo_url = "https://admin5500.s3.eu-central-1.amazonaws.com"
with open("photos.pk", "rb") as f:
    photos = pickle.load(f)

data = pd.read_csv("photos.csv", header=None, index_col=0)
done = os.listdir("data")
photos_name = [p["name"] for p in photos]


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


dnames = [d.split("/")[-1] for d in data[5]]
print("Start loading photos", len(set(dnames) ^ set(done)))
for p in tqdm(list(data[5][1:])):
    load(p)

print("Start preparing photos:", len(done) - len(photos_name))
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tf_pose

for photo in tqdm(done):
    if photo in photos_name:
        continue
    try:
        coco_style = tf_pose.infer(f"/home/petr/Documents/Projects/Recognition/data/{photo}")
    except Exception as e:
        print(e, f"data/{photo}")
    photos.append({
        "name": photo,
        "points": coco_style,
        "shape": None
    })
    photos_name.append(photo)

with open("photo.pk", "wb") as f:
    pickle.dump(photos, f)

print("Complete", len(photos))
