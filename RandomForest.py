#Classification
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 40 different person
oli = fetch_olivetti_faces()

plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i], cmap="gray")
    plt.axis("off")
    
plt.show()

X= oli.data
y= oli.target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)

rl_clf = RandomForestClassifier(n_estimators = 100, random_state=42)
rl_clf.fit(X_train, y_train)

y_pred = rl_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

# %%
#Regression
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)

rf_reg = RandomForestRegressor(random_state= 42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("rmse : ", rmse)
