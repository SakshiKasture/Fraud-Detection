import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

#Load Dataset
df = pd.read_csv("creditcard.csv")
print(df.head())

# Check for missing data
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(['Class'], axis=1))

# Adding the 'Class' column back
x = pd.DataFrame(scaled_features, columns=df.columns[:-1])  # Excluding 'Class' as it is the target variable used in train and test
y = df['Class']  # Target variable

#The stratify argument is used to ensure that both the training and testing 
# sets have the same proportion of fraud (class 1) and non-fraud (class 0) cases.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) 

print("Data preprocessing complete.")


# Initialize IsolationForest
isolation_forest = IsolationForest(contamination=0.01, random_state=42)  # Set contamination to the fraction of outliers
outliers = isolation_forest.fit_predict(x_train)

# The result will be -1 for outliers (fraudulent) and 1 for normal transactions (i.e is -1 is mapped to 1(fraud) and 1 to 0(non-fraud))
y_train_outliers = np.where(outliers == -1, 1, 0)

# You can compare the detected outliers (fraud) with the true labels (y_train) to check performance
print(f"Number of outliers detected: {np.sum(y_train_outliers)}")

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

rf.fit(x_train, y_train)

# Make predictions
y_pred = rf.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Precision, Recall, and F1 Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)