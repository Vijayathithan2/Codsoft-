import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Load the dataset
df = pd.read_csv("C:/Users/AdHi/Downloads/IRIS.csv")

#Explore the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

#Encode the target labels (species)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])  # setosa=0, versicolor=1, virginica=2

#Split the data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#Predict and evaluate
y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Predict on a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example
predicted_class = le.inverse_transform(clf.predict(sample))
print("\nPredicted class for sample {}: {}".format(sample[0], predicted_class[0]))
