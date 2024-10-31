from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

tree_clf = DecisionTreeClassifier(criterion= "gini", max_depth=5, random_state= 42)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(conf_matrix)

plt.figure(figsize = (15,10))
plot_tree(tree_clf, filled = True, feature_names = iris.feature_names, class_names = iris.target_names)
plt.show()

feature_imp = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_imp_sorted = sorted(zip(feature_imp, feature_names), reverse = True)
for importance, feature_name in feature_imp_sorted:
    print(f"{feature_name}: {importance}")