import sklearn
from sklearn import neighbors
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn import metrics

x = np.load("./TestDataSkimage/data/X.npy", allow_pickle=True)
y = np.load("./TestDataSkimage/data/Y.npy", allow_pickle=True)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1) #delete random state for random results


# Generate list of accuracies, one per K-value, to compare to each other:
accuracies = []
for i in range(50,95):
    n_neighbors = i
    nn = neighbors.KNeighborsClassifier(n_neighbors)
    nn.fit(x_train,y_train)
    predictions = nn.predict(x_test)
    
    acc = 0
    for j in range(len(y_test)):
        # If the prediciton matches the actual class, add 1 to the accumulator:
        if y_test[j] == predictions[j]:
            acc += 1
    ###
    
    accuracies.append(round(acc/len(y_test), 3))
###

# Plot out chart for K-value vs Accuracy:
#plt.figure(figsize=(20,20))
plt.title("K-Value and Associated Model Accuracy")
plt.xlabel("Value of K")
plt.ylabel("Model Accuracy")
plt.plot([n for n in range(50,95)], accuracies, marker='o', linestyle='-')
plt.grid(True)
plt.savefig("./KNN_KVal_Comparison_.png")

'''
shortAcc = accuracies[69:]
plt.plot([n for n in range(70,95)], shortAcc, marker='o', linestyle='-')
plt.grid(True)
plt.savefig("./KNN_KVal_Comparison_2.png")
'''

# print(predictions)
# "Class Value = " + x_test[i] + "----" + 
'''for i in range(len(y_test)):
    print(f"true value: {y_test[i]} ---- prediction: {predictions[i]}")
'''
print(f"Length of y Test = {len(y_test)}")

############
# ACTUAL CALCULAITONS/PREDICITONS BELOW
#############
GENUINE = 0

# Predict:
n_neighbors = 89
nn = neighbors.KNeighborsClassifier(n_neighbors)
nn.fit(x_train,y_train)
predictions = nn.predict(x_test)

# Accuracy:

acc = 0
for i in range(len(y_test)):
    # If the prediciton matches the actual class, add 1 to the accumulator:
    if y_test[i] == predictions[i]:
        acc += 1
###
print(f"Accuracy: {round(acc/len(y_test), 3)}")

#####

acc = 0
numOfPos = len([c for c in y_test if c==GENUINE])
for i in range(len(y_test)):
    # For every genuine signature that was correctly classified, add 1 to the accumulator:
    if y_test[i] == GENUINE and predictions[i] == GENUINE:
        acc += 1
###
print(f"Recall: {round(acc/numOfPos, 3)}")
numOfPosClassif = len([c for c in predictions if c==GENUINE])
print(f"Precision: {round(acc/numOfPosClassif, 3)}")

confusionMatrix = metrics.confusion_matrix(y_test, predictions)
confMatrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels = ["Gen", "Forg"])
confMatrix_display.plot()
plt.savefig("./confMatrix.png")

print("Confusion matrix eported into PNG")
