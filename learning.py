import collections
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

with open("photos.pk", "rb") as f:
    photos = pickle.load(f)

table = pd.read_csv("photos.csv", header=None)
counts = collections.Counter(list(table[8]))

photos = {p["name"]: p for p in photos}
data = table.to_dict(orient="record")
print(f"Find classes: {len(set(data[7]))}")

data = [{
    "photo": d[5].split("/")[-1],
    "result": d[7],
} for d in data]

res = []
for d in data:
    if d['photo'] in photos:
        d.update(photos[d['photo']])
        if d["result"] is None:
            continue
        if counts[d["result"]] < 30:
            d["result"] = "Прочее"
            print("Skip class")
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
print("labels:",len(labels))
X = np.array([clear(r["points"][0][0]) for r in res])
Y = np.array([labels.index(r["result"]) for r in res])

sm = SMOTE()
X, Y = sm.fit_sample(X, Y)

X_train, X_validation, y_train, y_validation = train_test_split(X, Y, train_size=0.95, random_state=12)
print(X_train.shape, X_validation.shape)

model = CatBoostClassifier(
    thread_count=3,
    iterations=1500,
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


with open("model.pk","wb") as f:
    pickle.dump(model, f)

with open("labels.pk","wb") as f:
    pickle.dump(labels, f)
