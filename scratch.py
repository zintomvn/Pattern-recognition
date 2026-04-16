import os
os.environ['KAGGLE_KEY'] = 'KGAT_f6214fc72120170c1ae1199574a5affc'
# Note: KAGGLE_USERNAME might literally not be needed for KGAT, let's test.
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

file_name = "CelebA_Spoof/Data/test/10001/live/498889.png"
print(f"Downloading {file_name}")
api.dataset_download_file("mabdullahsajid/celeba-spoofing", file_name=file_name, path="./test_dl")
print("Done.")
