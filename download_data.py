import os
import requests

def download_file(url, target_path):
    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"File downloaded successfully to {target_path}")

if __name__ == "__main__":
    dataset_url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    target_file = os.path.join(data_dir, "creditcard.csv")
    
    if not os.path.exists(target_file):
        download_file(dataset_url, target_file)
    else:
        print(f"File {target_file} already exists. Skipping download.")
