import zipfile
import os

def unzip_data(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")
    
    # List contents to verify
    print(f"Contents of {extract_to}:")
    for root, dirs, files in os.walk(extract_to):
        level = root.replace(extract_to, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        # Limit file listing
        for f in files[:5]:
            print(f'{subindent}{f}')
        if len(files) > 5:
            print(f'{subindent}... ({len(files)-5} more files)')

if __name__ == "__main__":
    zip_path = "archive.zip"
    extract_to = "data"
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
    else:
        os.makedirs(extract_to, exist_ok=True)
        unzip_data(zip_path, extract_to)
