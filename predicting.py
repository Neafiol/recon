import collections
import configparser
import json
import pickle

import numpy as np
import pandas as pd

config = configparser.RawConfigParser()
config.read('config.conf')

with open("model.pk", "rb") as f:
    model = pickle.load(f)
with open("photos.pk", "rb") as f:
    photos = pickle.load(f)
with open("labels.pk", "rb") as f:
    labels = pickle.load(f)

data = []
with open("input.jsonlines", "r") as f:
    for line in f.readlines():
        if line[-1] == "\n":
            line = line[:-1]
        dat = json.loads(line)
        dat["photo"] = dat["photo"].split("/")[-1]
        data.append(dat)

photos = {p["name"]: p for p in photos}

for_predict = []
for d in data:
    if d['photo'] in photos:
        d.update(photos[d['photo']])
        for_predict.append(d)

print(f"For predict: {len(for_predict)}/{len(data)}")


def clear(arr):
    res = []
    for i, k in enumerate(arr):
        if (i + 1) % 3:
            res.append(k)
    return res


X = np.array([clear(r["points"][0][0]) for r in for_predict])
y_pred = [model.predict(x) for x in X]

predicted_names = {}
for r, d in zip(y_pred, data):
    predicted_names[d["photo"]] = labels[int(r)]

with open("out.jsonlines", "w") as f:
    for d in data:
        if d["photo"] in predicted_names:
            f.write(json.dumps({"photo": d["photo"], "result": predicted_names[d["photo"]]}) + "\n")

print("file out.csv has been created")
