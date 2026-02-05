import os
import pandas as pd

# Update this to your local root path
BASE_DIRECTORY = r'D:\prec machine\datasets'

COLUMN_NAMES = {
    'acc': ['hour', 'minute', 'second', 'microsecond', 'horizontal_vibration', 'vertical_vibration'],
    'temp': ['hour', 'minute', 'second', 'decisecond', 'temperature']
}

def detect_separator_and_header(file_path):
    """
    Checks if a header exists and detects if the separator is ';' or ','.
    Returns (has_header, separator)
    """
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            
            # Check for existing header
            header_exists = 'hour' in first_line.lower()
            
            # Detect separator
            if ';' in first_line:
                sep = ';'
            else:
                sep = ','
                
            return header_exists, sep
    except Exception:
        return False, ','

def process_datasets(root_path):
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            if not filename.endswith('.csv'):
                continue
                
            file_path = os.path.join(root, filename)
            
            # Determine column names based on filename
            if filename.startswith('acc'):
                cols = COLUMN_NAMES['acc']
                file_type = "Vibration"
            elif filename.startswith('temp'):
                cols = COLUMN_NAMES['temp']
                file_type = "Temperature"
            else:
                continue

            # Check existing status
            header_exists, detected_sep = detect_separator_and_header(file_path)

            if header_exists:
                print(f"Skipping (Already Processed): {filename} in {os.path.basename(root)}")
                continue

            print(f"Processing {file_type} (Sep: '{detected_sep}'): {filename} in {os.path.basename(root)}")
            
            try:
                # Read data using the detected separator
                df = pd.read_csv(file_path, header=None, sep=detected_sep, engine='python')
                
                # Check if the split worked correctly
                if len(df.columns) == len(cols):
                    df.columns = cols
                    # Save it back as a standard comma-separated file with a header
                    df.to_csv(file_path, index=False, sep=',')
                else:
                    print(f"!!! Error in {filename}: Found {len(df.columns)} cols, expected {len(cols)}. Check file format.")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    if os.path.exists(BASE_DIRECTORY):
        print("Starting Dataset Correction Script...")
        process_datasets(BASE_DIRECTORY)
        print("\nAll files checked and updated.")
    else:
        print(f"Path not found: {BASE_DIRECTORY}")