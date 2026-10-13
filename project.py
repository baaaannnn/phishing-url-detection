import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import parallel_coordinates

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("--- Task 1: Attribute Extraction ---")

df = pd.read_csv("URLDataset.csv")

url_col = "URL"
label_col = "ClassLabel"

df["url_length"] = df[url_col].apply(lambda x: len(str(x)))
df["qty_dots"] = df[url_col].apply(lambda x: str(x).count("."))
df["qty_hyphens"] = df[url_col].apply(lambda x: str(x).count("-"))
df["qty_at"] = df[url_col].apply(lambda x: str(x).count("@"))

print(
df[
[
url_col,
"url_length",
"qty_dots",
"qty_hyphens",
"qty_at",
label_col
]
].head()
)

features = [
"url_length",
"qty_dots",
"qty_hyphens",
"qty_at"
]

scaler = MinMaxScaler()

df[features] = scaler.fit_transform(
df[features]
)

print("\n--- Task 2: Attribute Normalization ---")

print(
df[
features
].head()
)

print("\n--- Task 3: Attribute Visualization ---")

plot_df = df[
features + [label_col]
].copy()

plt.figure(figsize=(10, 6))

parallel_coordinates(
plot_df,
label_col,
color=["green", "red"]
)

plt.title("Parallel Coordinates Visualization")
plt.xlabel("Attributes")
plt.ylabel("Normalized Values")

plt.show()

print("\n--- Task 4: Build Classification Models ---")

X = df[features]
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
train_size=0.66,
random_state=42
)

print("Training Size:", len(X_train))
print("Testing Size:", len(X_test))

dt_model = DecisionTreeClassifier(
random_state=42
)

dt_model.fit(
X_train,
y_train
)

print("Decision Tree trained successfully")

knn_model = KNeighborsClassifier(
n_neighbors=7
)

knn_model.fit(
X_train,
y_train
)

print("K-NN (k=7) trained successfully")

print("\n--- Task 5: Decision Tree Visualization ---")

plt.figure(figsize=(16,10))

plot_tree(
dt_model,
feature_names=features,
class_names=dt_model.classes_,
filled=True
)

plt.title("Decision Tree Visualization")

plt.show()

print("\n--- Task 6: Model Accuracy Calculation ---")

dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

print("\nDecision Tree")

print(
"Accuracy:",
accuracy_score(
y_test,
dt_pred
)
)

print(
"Precision:",
precision_score(
y_test,
dt_pred,
pos_label="phishing"
)
)

print(
"Recall:",
recall_score(
y_test,
dt_pred,
pos_label="phishing"
)
)

print(
"F-measure:",
f1_score(
y_test,
dt_pred,
pos_label="phishing"
)
)

print("\nK-NN")

print(
"Accuracy:",
accuracy_score(
y_test,
knn_pred
)
)

print(
"Precision:",
precision_score(
y_test,
knn_pred,
pos_label="phishing"
)
)

print(
"Recall:",
recall_score(
y_test,
knn_pred,
pos_label="phishing"
)
)

print(
"F-measure:",
f1_score(
y_test,
knn_pred,
pos_label="phishing"
)
)

print("\n--- Task 7: Model Comparison ---")

dt_accuracy = accuracy_score(
y_test,
dt_pred
)

knn_accuracy = accuracy_score(
y_test,
knn_pred
)

if dt_accuracy > knn_accuracy:

    print(
    "\nDecision Tree performs better than K-NN."
    )

    print(
    "Reason: Decision Tree achieved higher classification accuracy and builds rule-based decisions from extracted URL attributes."
    )

elif knn_accuracy > dt_accuracy:

    print(
    "\nK-NN performs better than Decision Tree."
    )

    print(
    "Reason: K-NN achieved higher classification accuracy by classifying URLs based on similarity between feature values."
    )

else:

    print(
    "\nDecision Tree and K-NN achieved equal performance."
    )

    print(
    "Reason: Both models produced similar evaluation results."
    )