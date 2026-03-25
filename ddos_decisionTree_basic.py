import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv('dataset_sdn.csv')

# ── 2. Convert Protocol column to numbers ────────────────────
df['Protocol'] = pd.factorize(df['Protocol'])[0]

# ── 3. Prepare Features & Label ──────────────────────────────
X = df.drop(columns=['label', 'src', 'dst'])  # Features
y = df['label']                                # 0 = Normal, 1 = Attack

# ── 4. Split 80% Train, 20% Test ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 5. Train the Decision Tree ───────────────────────────────
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel training complete!")

# ── 6. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# ── 7. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("\nConfusion matrix saved as confusion_matrix.png!")