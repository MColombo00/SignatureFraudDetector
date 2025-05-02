import sklearn
from sklearn import neighbors
import numpy as np
import sklearn.model_selection

x = np.load("./TestDataSkimage/data/X.npy", allow_pickle=True)
y = np.load("./TestDataSkimage/data/Y.npy", allow_pickle=True)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)

nn = neighbors.KNeighborsClassifier(n_neighbors=5)
nn.fit(x_train,y_train)
predictions = nn.predict(x_test)

# print(predictions)
# "Class Value = " + x_test[i] + "----" + 
for i in range(len(y_test)):
    print(f"true value: {y_test[i]} ---- prediction: {predictions[i]}")
print(f"Length of y Test = {len(y_test)}")
