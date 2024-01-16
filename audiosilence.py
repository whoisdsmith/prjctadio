import os
import shutil
import logging
import argparse
import hashlib
import concurrent.futures
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser(description='Modify script to include new features.')
parser.add_argument('--silence_threshold', type=int, default=-50,
                    help='Silence threshold in dB')
parser.add_argument('--min_silence_length', type=int, default=1000,
                    help='Minimum length of silence at the beginning of the track in milliseconds')
parser.add_argument('--log_level', type=str, default='INFO',
                    help='Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)')
parser.add_argument('--threads', type=int, default=4,
                    help='Number of threads to use for concurrent processing')

args = parser.parse_args()

# Configuration
source_directory = "E:\\Instrumental Library"
destination_directory = os.path.join(source_directory, "AudioSilence")
silence_threshold = args.silence_threshold
min_silence_length = args.min_silence_length
log_file = "C:\\Users\\whois\\Documents\\prjctadio\\silence_detection.log"
processed_files_db = "processed_files.txt"
num_threads = args.threads

# Set up logging to both file and console
logging_level = getattr(logging, args.log_level.upper(), logging.INFO) # Fallback to INFO if level is not recognized
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging_level)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging_level)
console_handler.setFormatter(log_formatter)

# Configure the root logger
logging.getLogger().setLevel(logging_level)
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Read the list of already processed files and their checksums
processed_files = {}
if os.path.exists(processed_files_db):
    with open(processed_files_db, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                path, checksum = parts
                processed_files[path] = checksum

def get_file_checksum(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def file_exists(file_path):
    return os.path.isfile(file_path)

def file_analysis(file):
    if not file_exists(file):
        logging.warning(f"File {file} not found.")
        return None, None, None
    try:
        # Calculate file checksum
        checksum = get_file_checksum(file)
        
        # Skip processing if file has been processed and is unchanged
        if file in processed_files and processed_files[file] == checksum:
            logging.info(f"Skipping {file}, already processed and unchanged.")
            return None, None, None
        
        # Load the audio file
        try:
            audio = AudioSegment.from_file(file, format="mp3")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            return None, None, None
        
        # Check for non-silent parts
        nonsilent_parts = detect_nonsilent(audio, min_silence_len=min_silence_length, silence_thresh=silence_threshold)
        # If the first non-silent part starts after 0, there's silence at the beginning
        if nonsilent_parts and nonsilent_parts[0][0] > 0:
            return file, nonsilent_parts[0][0], checksum
        return None, None, None
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return None, None, None

def move_file(file, checksum):
    try:
        # Define the new path for the file
        dst_path = os.path.join(destination_directory, os.path.basename(file))
        # Move the file
        shutil.move(file, dst_path)
        logging.info(f"Moved: {file} -> {dst_path}")
        return True
    except Exception as e:
        logging.error(f"Error moving {file}: {e}")
        return False
    
def process_files(files):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {executor.submit(file_analysis, file): file for file in files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files), desc="Analyzing MP3 files", unit="file"):
            file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    file, silence_duration, checksum = result
                    if move_file(file, checksum):
                        with open(processed_files_db, 'a') as db:
                            db.write(f"{file},{checksum}\n")
                    else:
                        logging.info(f"No action required for {file} (file moved or no silence detected).")
                else:
                    logging.info(f"No result for {file} (silence not detected or file analysis failed).")
            except Exception as e:
                logging.error(f"Error with file {file}: {e}")

# Main execution code
def main():
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_directory) for f in filenames if f.endswith('.mp3')]
    process_files(files)

if __name__ == "__main__":
    main()