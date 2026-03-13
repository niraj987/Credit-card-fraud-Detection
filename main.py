import pandas as pd
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
import os

# Set plotting style
sns.set_theme(style="whitegrid")

# Create output directories
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def perform_eda(df):
    print("\n--- Exploratory Data Analysis ---")
    print(f"Missing values:\n{df.isnull().sum().max()}") # Usually 0 in this dataset
    
    # Class imbalance plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.close()
    
    # Amount distribution
    plt.figure(figsize=(10, 4))
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='b', alpha=0.5, label='Normal', kde=True)
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='r', alpha=0.5, label='Fraud', kde=True)
    plt.xlim(0, 2000)
    plt.legend()
    plt.title('Transaction Amount Distribution by Class')
    plt.savefig(f"{output_dir}/amount_distribution.png")
    plt.close()

def preprocess_data(df):
    print("\n--- Preprocessing Data ---")
    # Drop rows with missing values just in case
    df = df.dropna()
    
    # Scale Time and Amount
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    
    # Drop original Time and Amount columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Move scaled columns to the front for easier viewing (optional)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, X, y

def evaluate_model(y_true, y_pred, y_probs, model_name):
    print(f"\n--- Evaluation for {model_name} ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    
    if y_probs is not None:
        roc_auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(12, 5))
        
        # ROC Curve Plot
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend()
        
        # PR Curve Plot
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').lower()}_curves.png")
        plt.close()

def build_and_train_models(X_train, X_test, y_train, y_test, X, y):
    print("\n--- Model Training & Evaluation Setup ---")
    
    # Using Subsample for faster Random Forest Training
    print("\n1. Addressing Class Imbalance using SMOTE & Undersampling for Logistic Regression")
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    
    log_reg_pipeline = ImbPipeline(steps=[('smote', smote), ('under', under), ('classifier', LogisticRegression(max_iter=1000))])
    
    print("Training Logistic Regression with SMOTE...")
    log_reg_pipeline.fit(X_train, y_train)
    y_pred_lr = log_reg_pipeline.predict(X_test)
    y_probs_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_lr, y_probs_lr, "Logistic Regression (SMOTE)")

    print("\n2. Training Random Forest Classifier (Class Weight balanced)")
    # Using class_weight='balanced' instead of full SMOTE for RF to save time/memory, it often works well
    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_probs_rf = rf.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_pred_rf, y_probs_rf, "Random Forest")
    
    # Feature Importance for Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importances.png")
    plt.close()

    print("\n3. Training Isolation Forest (Anomaly Detection)")
    # Isolation forest is an unsupervised model, we train mostly on normal data, though it can take mixed data
    # We will pass the full training set
    iso_forest = IsolationForest(n_estimators=100, max_samples=len(X_train), contamination=float(y_train.mean()), random_state=42)
    print("Training Isolation Forest...")
    iso_forest.fit(X_train)
    
    # Predictions: 1 for normal, -1 for anomaly
    y_pred_iso_raw = iso_forest.predict(X_test)
    # Convert to 0 normal, 1 anomaly
    y_pred_iso = np.where(y_pred_iso_raw == 1, 0, 1)
    
    # Isolation forest provides anomaly scores (negative values, lower is more anomalous)
    y_probs_iso = -iso_forest.score_samples(X_test)
    
    evaluate_model(y_test, y_pred_iso, y_probs_iso, "Isolation Forest")

def main():
    data_path = "data/creditcard.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please run download_data.py first.")
        return
        
    df = load_data(data_path)
    perform_eda(df)
    
    X_train, X_test, y_train, y_test, X, y = preprocess_data(df)
    build_and_train_models(X_train, X_test, y_train, y_test, X, y)
    
    print(f"\nPipeline complete! Output plots saved to '{output_dir}/' directory.")

if __name__ == "__main__":
    main()
