import os
import requests
from concurrent.futures import ThreadPoolExecutor

# Your Zenodo API Access Token
ACCESS_TOKEN = "snjNr3qMUZCa4WuKBXr9UuyyvsDonMIt4ZQQXdyyY92mb9eBV5TOQPkODlWc"

# Zenodo record ID (change this to your actual record ID)
RECORD_ID = "6406114"

# Zenodo API endpoint for file listing
FILES_API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"

# Destination folder for downloads
SAVE_DIR = '/Volumes/T7 Shield/radchest'
os.makedirs(SAVE_DIR, exist_ok=True)

# Zenodo API endpoint for file listing
FILES_API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# Fetch list of files
response = requests.get(FILES_API_URL, headers=headers)
if response.status_code == 200:
    files = response.json().get("files", [])
else:
    print(f"❌ Failed to retrieve file list - Status Code {response.status_code}")
    exit(1)

downloaded_files = [file for file in os.listdir(SAVE_DIR) if not file.startswith('._')]
files = [file for file in files if file['key'] not in downloaded_files]

# Define the file download function
def download_file(file):
    file_name = file["key"]
    file_url = file["links"]["self"]
    file_path = os.path.join(SAVE_DIR, file_name)
    
    print(f"📥 Starting download: {file_name}")

    # Large chunk size for efficiency
    CHUNK_SIZE = 4 * 1024 * 1024  # 4MB

    try:
        with requests.get(file_url, headers=headers, stream=True) as file_response:
            if file_response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in file_response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Downloaded: {file_name}")
            else:
                print(f"❌ Failed to download: {file_name} - Status {file_response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading {file_name}: {e}")

# Use ThreadPoolExecutor for parallel downloads
NUM_WORKERS = min(5, len(files))  # Use up to 8 threads or the number of files
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    executor.map(download_file, files)

print("✅ All downloads complete!")