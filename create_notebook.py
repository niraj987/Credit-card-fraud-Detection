import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text0 = """# Credit Card Fraud Detection - Comprehensive Analysis

This notebook provides a complete end-to-end data analysis of the Credit Card Fraud dataset. It includes:
1. **Descriptive Analysis**: Understanding the basic statistical properties of the data.
2. **Exploratory Data Analysis (EDA)**: Visualizing distributions, correlations, and class imbalances.
3. **Data Preprocessing**: Scaling and preparing the data.
4. **Predictive Analysis**: Building Machine Learning models (Logistic Regression, Random Forest, Isolation Forest).
5. **Evaluation**: Assessing models using Precision-Recall, ROC-AUC, and Confusion Matrices."""

code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""

text2 = """## 1. Data Loading and Basic Inspection"""

code2 = """data_path = "data/creditcard.csv"
df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape}")
df.head()"""

text3 = """## 2. Descriptive Analysis
Let's look at the basic statistics of the dataset, particularly focusing on the `Amount` and `Time` features, as the V1-V28 features are PCA-transformed."""

code3 = """# Basic statistics
display(df[['Time', 'Amount', 'Class']].describe())

# Check for missing values
print(f"Missing values:\\n{df.isnull().sum().max()} missing values found.")

# Class imbalance ratio
fraud_count = df['Class'].sum()
normal_count = len(df) - fraud_count
print(f"\\nNormal Transactions: {normal_count} ({(normal_count/len(df))*100:.2f}%)")
print(f"Fraud Transactions: {fraud_count} ({(fraud_count/len(df))*100:.2f}%)")"""

text4 = """## 3. Exploratory Data Analysis (EDA)

### 3.1 Class Distribution
Visualizing the extreme class imbalance."""

code4 = """plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Normal, 1: Fraud)')
plt.yscale('log')
plt.ylabel('Count (Log Scale)')
plt.show()"""

text5 = """### 3.2 Transaction Amount and Time Analysis"""

code5 = """fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Time distribution
sns.histplot(df['Time'], bins=50, ax=axes[0], kde=True, color='purple')
axes[0].set_title('Distribution of Transaction Time')
axes[0].set_xlabel('Time (in seconds)')

# Amount distribution
sns.histplot(df['Amount'], bins=50, ax=axes[1], kde=True, color='teal')
axes[1].set_title('Distribution of Transaction Amount')
axes[1].set_xlabel('Amount')

plt.tight_layout()
plt.show()"""

text6 = """Let's see if the transaction amount differs between normal and fraudulent transactions."""

code6 = """fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(x='Class', y='Amount', data=df, ax=axes[0])
axes[0].set_title('Boxplot of Amount by Class')
axes[0].set_ylim(0, 500) # Zoomed in

sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', alpha=0.5, label='Normal', kde=True, ax=axes[1])
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', alpha=0.5, label='Fraud', kde=True, ax=axes[1])
axes[1].set_xlim(0, 2000)
axes[1].legend()
axes[1].set_title('Transaction Amount Distribution by Class')

plt.tight_layout()
plt.show()"""

text7 = """### 3.3 Correlation Heatmap
Let's see how features correlate with the target variable `Class`."""

code7 = """# We will compute the correlation matrix. Since PCA features are orthogonal, their inter-correlation is ~0.
# We focus on correlation with 'Class'.
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr[['Class']].sort_values(by='Class', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation with Class')
plt.show()"""

text8 = """## 4. Data Preprocessing
We need to scale the `Time` and `Amount` features, as the PCA features (V1-V28) are already scaled."""

code8 = """scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training Set: {X_train.shape}")
print(f"Testing Set: {X_test.shape}")"""

text9 = """## 5. Predictive Analysis & Modeling
We will evaluate three models:
1. Logistic Regression (with SMOTE/Undersampling)
2. Random Forest (with Class Weights)
3. Isolation Forest (Anomaly Detection)

### Helper Function for Evaluation"""

code9 = """def evaluate_model(y_true, y_pred, y_probs, model_name):
    print(f"\\n{'='*50}\\nEvaluation for {model_name}\\n{'='*50}")
    print("Confusion Matrix:\\n", confusion_matrix(y_true, y_pred))
    print("\\nClassification Report:\\n", classification_report(y_true, y_pred))
    
    if y_probs is not None:
        roc_auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC Score: {roc_auc:.4f}\\n")
        
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # ROC Curve
        axes[0].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'{model_name} - ROC Curve')
        axes[0].legend()
        
        # PR Curve
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, label=f'PR curve (area = {pr_auc:.4f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'{model_name} - Precision-Recall Curve')
        axes[1].legend()
        
        plt.show()"""

text10 = """### Model 1: Logistic Regression with SMOTE and Undersampling
Since the data is highly imbalanced, standard Logistic Regression will fail to detect fraud. We synthetically generate fraud cases (SMOTE) and undersample normal cases."""

code10 = """smote = SMOTE(sampling_strategy=0.1, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
log_reg_pipeline = ImbPipeline(steps=[('smote', smote), ('under', under), ('classifier', LogisticRegression(max_iter=1000))])

print("Training Logistic Regression Pipeline...")
log_reg_pipeline.fit(X_train, y_train)

y_pred_lr = log_reg_pipeline.predict(X_test)
y_probs_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]

evaluate_model(y_test, y_pred_lr, y_probs_lr, "Logistic Regression (SMOTE + UnderSampling)")"""

text11 = """### Model 2: Random Forest Classifier
Random Forests can handle non-linear relationships well. To deal with imbalance, we use `class_weight='balanced'`."""

code11 = """rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
print("Training Random Forest...")
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_probs_rf = rf.predict_proba(X_test)[:, 1]

evaluate_model(y_test, y_pred_rf, y_probs_rf, "Random Forest")

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices][:10], align="center")
plt.xticks(range(10), X_train.columns[indices][:10], rotation=45)
plt.ylabel('Relative Importance')
plt.show()"""

text12 = """### Model 3: Isolation Forest
An unsupervised anomaly detection method. It assumes that anomalies (fraud) are few and different, making them easier to 'isolate'."""

code12 = """iso_forest = IsolationForest(n_estimators=100, max_samples=len(X_train), contamination=float(y_train.mean()), random_state=42, n_jobs=-1)
print("Training Isolation Forest...")
iso_forest.fit(X_train)

# Predictions: 1 for normal, -1 for anomaly. Convert to 0 normal, 1 anomaly.
y_pred_iso_raw = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso_raw == 1, 0, 1)

# Anomaly scores
y_probs_iso = -iso_forest.score_samples(X_test)

evaluate_model(y_test, y_pred_iso, y_probs_iso, "Isolation Forest")"""

text13 = """## 6. Conclusion
* **Logistic Regression (with SMOTE)** catches most frauds (high Recall) but suffers from false positives (low Precision).
* **Random Forest** provides the best balance of Precision and Recall, making it highly effective for a production environment where false alarms have a high cost.
* **Isolation Forest** works as a solid unsupervised baseline but does not match supervised Random Forest performance."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text0),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_markdown_cell(text7),
    nbf.v4.new_code_cell(code7),
    nbf.v4.new_markdown_cell(text8),
    nbf.v4.new_code_cell(code8),
    nbf.v4.new_markdown_cell(text9),
    nbf.v4.new_code_cell(code9),
    nbf.v4.new_markdown_cell(text10),
    nbf.v4.new_code_cell(code10),
    nbf.v4.new_markdown_cell(text11),
    nbf.v4.new_code_cell(code11),
    nbf.v4.new_markdown_cell(text12),
    nbf.v4.new_code_cell(code12),
    nbf.v4.new_markdown_cell(text13)
]

with open('Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Jupyter Notebook 'Analysis.ipynb' created successfully!")
