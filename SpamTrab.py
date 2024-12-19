import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load the dataset
data_path = 'D:/MIAA/2-Ano/CiberSegurança/Trabalho/spambase/spambase.csv'  # Replace with your dataset path
data = pd.read_csv(data_path)

# Preview the dataset
print("Dataset Overview:")
print(data.head())

# Data Preprocessing
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Split data into features and target variable
X = data.drop('spam', axis=1)  # Features
y = data['spam']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training and Evaluation
# Logistic Regression
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, log_reg_preds))
print(classification_report(y_test, log_reg_preds))

# Support Vector Machine (SVM)
print("\nTraining Support Vector Machine (SVM)...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
print("SVM Results:")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# Results Visualization
# Confusion Matrix for Logistic Regression
log_reg_cm = confusion_matrix(y_test, log_reg_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for SVM
svm_cm = confusion_matrix(y_test, svm_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC for Logistic Regression
log_reg_prob = log_reg.predict_proba(X_test)[:, 1]
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg_prob)
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

plt.figure(figsize=(10, 6))
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')

# ROC Curve and AUC for SVM
svm_prob = svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Compare Models
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM'],
    'Accuracy': [accuracy_score(y_test, log_reg_preds), accuracy_score(y_test, svm_preds)],
    'AUC': [roc_auc_log_reg, roc_auc_svm]
})

print("\nModel Comparison:")
print(results)

# Conclusion
print("\nConclusion:")
if results.loc[0, 'AUC'] > results.loc[1, 'AUC']:
    print("Logistic Regression demonstrates better performance in terms of AUC and is more suitable for this spam detection problem.")
elif results.loc[0, 'AUC'] < results.loc[1, 'AUC']:
    print("SVM demonstrates better performance in terms of AUC and is more suitable for this spam detection problem.")
else:
    print("Both models perform similarly in terms of AUC. Further exploration with additional features or tuning might be necessary.")
