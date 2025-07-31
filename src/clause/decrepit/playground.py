from imodels import RuleFitClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
import pandas as pd

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=5,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

print(X)
print(y)

model = OneVsRestClassifier(RuleFitClassifier(
    n_estimators=25,
    tree_size=3
))
model.fit(X, y)

print("Training Done")

import time

now = time.time()

y_pred = model.predict(X)
print(classification_report(y, y_pred))

speed = time.time() - now

print("Time taken for prediction:", speed)
print("Time taken for average sentence:", speed * 17.5)

for i, clf in enumerate(model.estimators_):
    print(f"Class {i} rules:")
    for rule in clf.rules_:
        print(rule)
    print()

print(model.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]))