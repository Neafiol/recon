import collections
import pickle

import numpy as np
import pandas as pd

with open("model.pk", "rb") as f:
    model = pickle.load(f)
with open("photos.pk", "rb") as f:
    photos = pickle.load(f)
with open("labels.pk", "rb") as f:
    labels = pickle.load(f)

RESULT_COL = 8
table = pd.read_csv("photos.csv", header=None)
counts = collections.Counter(list(table[RESULT_COL]))


photos = {p["name"]: p for p in photos}
data = table.to_dict(orient="record")
data = [{
    "photo": d[5].split("/")[-1],
    "result": d[RESULT_COL],
} for d in data]

for_predict = []
for d in data:
    if d['photo'] in photos:
        d.update(photos[d['photo']])
        if d["result"] not in labels:
            if len(d["points"]):
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

full_table = table.to_dict(orient="record")
for t in full_table:
    name = t[5].split("/")[-1]
    if name in predicted_names:
        t[RESULT_COL] = predicted_names[name]

csv = pd.DataFrame.from_dict(full_table)
csv.to_csv("fphotos.csv",index=False)
print("file fphotos.csv has been created")
