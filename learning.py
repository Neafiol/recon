import collections
import configparser
import json
import pickle

import numpy as np
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

with open("photos.pk", "rb") as f:
    photos = pickle.load(f)

config = configparser.RawConfigParser()
config.read('config.conf')
PHOTO_COL = int(config.get("SETTING", "photo_url"))
RESULT_COL = int(config.get("SETTING", "result"))

photos = {p["name"]: p for p in photos}
data = []
with open("learning.jsonlines", "r") as f:
    for line in f.readlines():
        if line[-1] == "\n":
            line = line[:-1]
        dat = json.loads(line)
        dat["photo"] = dat["photo"].split("/")[-1]
        data.append(dat)

counts = collections.Counter([d["result"] for d in data])
print(f"Find classes: {len(counts)}")
assert len(counts) > 0, "only one class found"


res = []
for d in data:
    if d['photo'] in photos:
        d.update(photos[d['photo']])
        if counts[d["result"]] < 30:
            print("Skip class", d["result"], counts[d["result"]])
            d["result"] = "Прочее"
        if len(d["points"]):
            res.append(d)

print(f"Dataset: {len(res)}/{len(data)}")


def clear(arr):
    res = []
    for i, k in enumerate(arr):
        if (i + 1) % 3:
            res.append(k)
    return res


labels = list(set([r["result"] for r in res]))
print("labels:", len(labels))
X = np.array([clear(r["points"][0][0]) for r in res])
Y = np.array([labels.index(r["result"]) for r in res])

sm = SMOTE()
X, Y = sm.fit_sample(X, Y)

X_train, X_validation, y_train, y_validation = train_test_split(X, Y, train_size=0.95, random_state=12)
print(X_train.shape, X_validation.shape)

model = CatBoostClassifier(
    thread_count=3,
    iterations=1600,
    depth=6,
    use_best_model=True
)
model.fit(
    X_train, y_train,
    eval_set=(X_validation, y_validation),
    # logging_level='Silent',
    # plot=False
)

from sklearn import metrics

y_pred = [model.predict(x) for x in X_validation]
print(metrics.classification_report(y_validation, y_pred,
                                    digits=3, target_names=labels))

with open("model.pk", "wb") as f:
    pickle.dump(model, f)

with open("labels.pk", "wb") as f:
    pickle.dump(labels, f)
