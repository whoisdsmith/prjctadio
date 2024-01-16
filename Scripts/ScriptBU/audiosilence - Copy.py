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

args = parser.parse_args()

# Configuration
source_directory = "E:\\Instrumental Library"
destination_directory = os.path.join(source_directory, "AudioSilence")
silence_threshold = args.silence_threshold
min_silence_length = args.min_silence_length
log_file = "C:\\Users\\whois\\Documents\\prjctadio\\silence_detection.log"
processed_files_db = "processed_files.txt"

# Set up logging
logging_level = getattr(logging, args.log_level.upper(), None)
logging.basicConfig(filename=log_file, level=logging_level,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Read the list of already processed files and their checksums
processed_files = {}
if os.path.exists(processed_files_db):
    with open(processed_files_db, 'r') as file:
        for line in file:
            path, checksum = line.strip().split(',')
            processed_files[path] = checksum

def get_file_checksum(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def file_analysis(file):
    try:
        # Calculate file checksum
        checksum = get_file_checksum(file)
        
        # Skip processing if file has been processed and is unchanged
        if file in processed_files and processed_files[file] == checksum:
            logging.info(f"Skipping {file}, already processed and unchanged.")
            return None, None
        
        # Load the audio file
        audio = AudioSegment.from_file(file, format="mp3")
        
        # Check for non-silent parts
        nonsilent_parts = detect_nonsilent(audio, min_silence_len=min_silence_length, silence_thresh=silence_threshold)
        # If the first non-silent part starts after 0, there's silence at the beginning
        if nonsilent_parts and nonsilent_parts[0][0] > 0:
            return file, nonsilent_parts[0][0], checksum
        return None, None, None
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return None, None, None

def move_file(file):
    try:
        # Define the new path for the file
        dst_path = os.path.join(destination_directory, os.path.basename(file))
        # Move the file
        shutil.move(file, dst_path)
        logging.info(f"Moved: {file} -> {dst_path}")
    except Exception as e:
        logging.error(f"Error moving {file}: {e}")

def process_files(files):
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(file_analysis, file): file for file in files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Analyzing MP3 files", unit="file"):
            file, silence_duration, checksum = future.result()
            if file:
                logging.info(f"Silence detected in {file}: {silence_duration}ms")
                move_file(file)
                with open(processed_files_db, 'a') as db:
                    db.write(f"{file},{checksum}\n")

# Start the process
logging.info("Starting to scan for silence in MP3 files.")

all_files = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(source_directory)
    for file in files if file.lower().endswith('.mp3')
]

# Process files
process_files(all_files)

logging.info("Scan and move complete.")