import pandas as pd

# Load Titanic dataset
df = pd.read_csv("C:/Users/AdHi/Downloads/Titanic-Dataset.csv")

# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# Manual label encoding
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(embarked_mapping)

# Manual train/test split (80% train, 20% test)
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# Define features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]
y_test = test_data['Survived']

# Simple model
def predict(row):
    if row['Sex'] == 0:  # female
        return 1
    elif row['Pclass'] == 1 and row['Age'] < 18:
        return 1
    else:
        return 0

# Generate predictions High precision = fewer false data
y_pred = X_test.apply(predict, axis=1)

# Metrics Calculation
# Accuracy
correct = (y_pred == y_test).sum()
total = len(y_test)
accuracy = correct / total

# Confusion Matrix
''' TP: Correctly predicted survived
    FP: Predicted survived but actually died
    TN: Correctly predicted died
    FN: Predicted died but actually survived'''
tp = ((y_pred == 1) & (y_test == 1)).sum()
tn = ((y_pred == 0) & (y_test == 0)).sum()
fp = ((y_pred == 1) & (y_test == 0)).sum()
fn = ((y_pred == 0) & (y_test == 1)).sum()

# Classification Report
precision = tp / (tp + fp) if (tp + fp) > 0 else 0 #Accuracy
recall = tp / (tp + fn) if (tp + fn) > 0 else 0 #High recall = fewer missed survivors
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 #A balance between Precision and Recall
'''f1 = 1 good precision and recall
   f1 = 0 worst prediction'''

#Output
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

print("\nClassification Report:")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
