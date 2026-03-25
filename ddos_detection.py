import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# -------------------------------------------------------
# STEP 1: Load Dataset
# -------------------------------------------------------
df = pd.read_csv('dataset_sdn.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nClass Distribution:\n{df['label'].value_counts()}")

# -------------------------------------------------------
# STEP 2: Preprocessing
# -------------------------------------------------------
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------------------------------
# STEP 3: Add Same Noise as Her (5% label flip)
# -------------------------------------------------------
np.random.seed(42)
noise_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[noise_idx, 'label'] = 1 - df.loc[noise_idx, 'label']

# -------------------------------------------------------
# STEP 4: Drop Same Features as Her
# -------------------------------------------------------
X = df.drop(columns=['pktrate', 'tot_kbps', 'tx_kbps', 'rx_kbps', 'label'])
y = df['label']

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape  : {y.shape}")

# -------------------------------------------------------
# STEP 5: Train/Test Split (70/30 like her)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# -------------------------------------------------------
# STEP 6: Train Decision Tree
# -------------------------------------------------------
print("\nTraining Decision Tree model...")
start_train = time.time()

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"Training Time: {train_time:.4f} seconds")

# -------------------------------------------------------
# STEP 7: Predict
# -------------------------------------------------------
start_pred = time.time()
y_pred = dt_model.predict(X_test)
pred_time = time.time() - start_pred
print(f"Prediction Time: {pred_time:.4f} seconds")

# -------------------------------------------------------
# STEP 8: Evaluation Metrics
# -------------------------------------------------------
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n" + "=" * 60)
print("DECISION TREE - EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy  : {accuracy  * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall    * 100:.2f}%")
print(f"F1 Score  : {f1        * 100:.2f}%")
print(f"Train Time: {train_time:.4f}s  |  Predict Time: {pred_time:.4f}s")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------
# STEP 9: Confusion Matrix
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'DDoS'],
            yticklabels=['Normal', 'DDoS'])
plt.title('Decision Tree - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('dt_confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved as 'dt_confusion_matrix.png'")

# -------------------------------------------------------
# STEP 10: Save Metrics Summary
# -------------------------------------------------------
results = {
    'Algorithm'     : ['Decision Tree'],
    'Accuracy (%)'  : [round(accuracy  * 100, 2)],
    'Precision (%)' : [round(precision * 100, 2)],
    'Recall (%)'    : [round(recall    * 100, 2)],
    'F1 Score (%)'  : [round(f1        * 100, 2)],
    'Train Time (s)': [round(train_time, 4)],
    'Pred Time (s)' : [round(pred_time,  4)],
}

results_df = pd.DataFrame(results)
results_df.to_csv('dt_results_summary.csv', index=False)
print("\nResults summary saved to 'dt_results_summary.csv'")
print(results_df.to_string(index=False))