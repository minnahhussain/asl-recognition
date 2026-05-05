import csv
import pickle
from sklearn.neighbors import KNeighborsClassifier

X = [] #features (63 nums)
Y = [] # labels(letters)

with open('hand_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        Y.append(row[0])
        X.append([float(val) for val in row[1:]])


model = KNeighborsClassifier(n_neighbors=4)
model.fit(X, Y)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(model.predict([X[0]]))
print(model.score(X, Y))



