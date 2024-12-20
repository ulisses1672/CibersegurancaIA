import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from time import time

# Load the dataset
data_path = 'D:/MIAA/2-Ano/CiberSegurança/Trabalho/spambase/spambase.csv' 
data = pd.read_csv(data_path)

# Preview the dataset
print("Dataset Overview:")
print(data.head())

# Data Preprocessing
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Scale the features for optimization
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('spam', axis=1))  # Features
y = data['spam']  # Target

# Subset Analysis by Feature Range
subsets = [
    ("Low Capital Run Length", data[data['capital_run_length_average'] < data['capital_run_length_average'].median()]),
    ("High Capital Run Length", data[data['capital_run_length_average'] >= data['capital_run_length_average'].median()])
]

subset_results = []
for subset_name, subset_data in subsets:
    print(f"\nAnalyzing Subset: {subset_name}")
    subset_X = scaler.fit_transform(subset_data.drop('spam', axis=1))
    subset_y = subset_data['spam']

    X_train, X_test, y_train, y_test = train_test_split(subset_X, subset_y, test_size=0.3, random_state=42)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=20000, solver='lbfgs', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)
    log_reg_preds = log_reg.predict(X_test)
    log_reg_auc = auc(*roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])[:2])

    # SVM
    svm = SVC(kernel='rbf', C=20, gamma='scale', probability=True, max_iter=20000)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_auc = auc(*roc_curve(y_test, svm.predict_proba(X_test)[:, 1])[:2])

    # Random Forest
    rf = RandomForestClassifier(n_estimators=1500, max_depth=60, random_state=42, criterion='gini')
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_auc = auc(*roc_curve(y_test, rf.predict_proba(X_test)[:, 1])[:2])

    # KNN
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance', metric='minkowski', p=2)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_auc = auc(*roc_curve(y_test, knn.predict_proba(X_test)[:, 1])[:2])

    # Collect Results
    subset_results.append({
        'Subset': subset_name,
        'Logistic Regression AUC': log_reg_auc,
        'SVM AUC': svm_auc,
        'Random Forest AUC': rf_auc,
        'KNN AUC': knn_auc
    })

# Display Subset Results
subset_results_df = pd.DataFrame(subset_results)
print("\nSubset Comparison Results:")
print(subset_results_df)

# Visualize Subset Results
subset_results_df.set_index('Subset').plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Across Subsets')
plt.ylabel('AUC')
plt.xlabel('Subset')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model', loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.show()


# General Model Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Timing and Training Models
timing_results = []

# Logistic Regression
print("\nTraining Logistic Regression...")
start_time = time()
log_reg = LogisticRegression(max_iter=20000, solver='lbfgs', penalty='l2', C=1.0)
log_reg.fit(X_train, y_train)
log_reg_time = time() - start_time
log_reg_preds = log_reg.predict(X_test)
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, log_reg_preds))
print(classification_report(y_test, log_reg_preds))
timing_results.append({'Model': 'Logistic Regression', 'Time': log_reg_time})

# Support Vector Machine (SVM)
print("\nTraining Support Vector Machine (SVM)...")
start_time = time()
svm = SVC(kernel='rbf', C=20, gamma='scale', probability=True, max_iter=20000)
svm.fit(X_train, y_train)
svm_time = time() - start_time
svm_preds = svm.predict(X_test)
print("SVM Results:")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))
timing_results.append({'Model': 'SVM', 'Time': svm_time})

# Random Forest
print("\nTraining Random Forest Classifier...")
start_time = time()
rf = RandomForestClassifier(n_estimators=1500, max_depth=60, random_state=42, criterion='gini')
rf.fit(X_train, y_train)
rf_time = time() - start_time
rf_preds = rf.predict(X_test)
print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
timing_results.append({'Model': 'Random Forest', 'Time': rf_time})

# K-Nearest Neighbors
print("\nTraining K-Nearest Neighbors (KNN)...")
start_time = time()
knn = KNeighborsClassifier(n_neighbors=20, weights='distance', metric='minkowski', p=2)
knn.fit(X_train, y_train)
knn_time = time() - start_time
knn_preds = knn.predict(X_test)
print("KNN Results:")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))
timing_results.append({'Model': 'KNN', 'Time': knn_time})

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

# Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for KNN
knn_cm = confusion_matrix(y_test, knn_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
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

# ROC Curve and AUC for Random Forest
rf_prob = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# ROC Curve and AUC for KNN
knn_prob = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)
roc_auc_knn = auc(fpr_knn, tpr_knn)
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Compare Models
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'KNN'],
    'Accuracy': [
        accuracy_score(y_test, log_reg_preds),
        accuracy_score(y_test, svm_preds),
        accuracy_score(y_test, rf_preds),
        accuracy_score(y_test, knn_preds)
    ],
    'AUC': [roc_auc_log_reg, roc_auc_svm, roc_auc_rf, roc_auc_knn]
})

print("\nModel Comparison:")
print(results)

# Timing Comparison
timing_df = pd.DataFrame(timing_results)
print("\nTiming Results:")
print(timing_df)

# Plot Timing Results
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Time', data=timing_df, palette='viridis')
plt.title('Training Time by Model')
plt.ylabel('Time (seconds)')
plt.xlabel('Model')
plt.show()

# Conclusion
print("\nConclusion:")
best_model = results.loc[results['AUC'].idxmax()]
print(f"The best performing model is {best_model['Model']} with an AUC of {best_model['AUC']:.2f}.")
