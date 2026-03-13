# Credit Card Fraud Detection Pipeline

## 📌 Project Overview
The **Credit Card Fraud Detection** project is an end-to-end Machine Learning pipeline designed to identify fraudulent credit card transactions. Credit card fraud is a massive problem in the Fintech, Banking, and Payments industries (e.g., Paytm, PhonePe, Razorpay). This project tackles the inherent **high-stakes class imbalance** problem, where fraudulent transactions are extremely rare compared to normal transactions.

## 🚀 Why This Project Matters (Future Impact)
As digital transactions grow, so do sophisticated fraudulent activities. This project provides a foundational blueprint for:
1. **Financial Security:** Protecting customers from unauthorized charges and securing bank assets from massive losses.
2. **Real-time Anomaly Detection:** The models developed here (like Random Forest and Isolation Forest) can be adapted for real-time streaming pipelines to block transactions instantly.
3. **Imbalanced Data Handling:** Techniques demonstrated here (SMOTE, Class Weights, Undersampling) are highly transferable to other rare-event detection domains like Medical Diagnosis (rare diseases), Network Intrusion Detection, and Manufacturing Defect Detection.

## 🛠️ Tech Stack & Tools Used
*   **Python:** Core programming language.
*   **Pandas & NumPy:** For Data Manipulation, Descriptive Analysis, and Preprocessing.
*   **Scikit-Learn:** Core library for Machine Learning models, Preprocessing (`StandardScaler`), and Evaluation Metrics.
*   **Imbalanced-Learn (imblearn):** For specialized sampling techniques (`SMOTE` for oversampling, `RandomUnderSampler`).
*   **Matplotlib & Seaborn:** For Exploratory Data Analysis (EDA) and visualizing model performance (ROC-AUC, Precision-Recall curves).
*   **Jupyter Notebook (`nbformat`, `nbconvert`):** For interactive data analysis, EDA, and Predictive Analysis.

## 📂 Project Structure
*   `download_data.py`: An automated script to fetch the raw dataset securely from GitHub.
*   `main.py`: The core pipeline script that handles data loading, preprocessing, model training, and saves evaluation visualizations to the `output/` folder.
*   `create_notebook.py`: A script that generates the Jupyter Notebook programmatically.
*   `Analysis.ipynb`: A comprehensive, fully executed notebook that contains Descriptive Analysis, Interactive EDA, Data Preprocessing, and Predictive Analysis step-by-step.
*   `requirements.txt`: List of all Python dependencies.
*   `output/`: Directory where all generated plots (EDA & Evaluation Curves) are saved.

## ⚙️ How to Run the Project

### Prerequisites
Make sure you have Python 3.8+ installed.

### 1. Install Dependencies
Clone or download the repository, then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
Run the data acquisition script to fetch the `creditcard.csv` dataset.
```bash
python download_data.py
```
> The dataset will be downloaded into a dynamically created `data/` folder.

### 3. Run the Main Pipeline
Execute the full, end-to-end training and evaluation script:
```bash
python main.py
```
> You will see precision, recall, and F1-score outputs in the terminal. ROC-AUC curves, Precision-Recall curves, and Feature Importance plots will be saved in the `output/` directory.

### 4. Interactive Analysis (Jupyter Notebook)
To view the detailed Descriptive Analysis, Exploratory Data Analysis (EDA), and Predictive Modeling interactively:
1. Open `Analysis.ipynb` in VS Code, JupyterLab, or your preferred IDE to view the executed results.
2. If you want to recreate the notebook from scratch, run `python create_notebook.py` and execute the cells.

## 🧠 Models Evaluated & Strategy
We trained and evaluated three distinct approaches to handling fraud:

1.  **Logistic Regression (with SMOTE and Undersampling):**
    *   **Strategy:** We synthetically generated fraud cases (SMOTE) and undersampled normal data to give the model a balanced view.
    *   **Result:** Extremely high **Recall** (catches almost all fraud), but lower Precision (high false-positive rate).
2.  **Random Forest Classifier:**
    *   **Strategy:** Utilized `class_weight='balanced'` within the ensemble to penalize misclassifications of the minority class heavily.
    *   **Result:** The best overall performer. It provides an excellent balance of high **Precision** (very few false alarms) and solid Recall, making it ideal for production where customer friction must be minimized.
3.  **Isolation Forest (Anomaly Detection):**
    *   **Strategy:** An unsupervised approach that isolates anomalies based on the assumption that frauds are "few and different."
    *   **Result:** A strong baseline that doesn't rely entirely on perfectly labeled historical fraud data.

## 📊 Key Evaluation Metrics
In fraud detection, standard "Accuracy" is misleading because 99.8% of transactions are normal. Instead, we focus on:
*   **Recall:** Out of all actual frauds, how many did we catch? (Crucial for stopping financial loss).
*   **Precision:** Out of all flagged frauds, how many were actually fraud? (Crucial for avoiding customer frustration with blocked cards).
*   **ROC-AUC & Precision-Recall Curves:** Used heavily in this project to visualize the trade-offs between precision and recall across different thresholds.
