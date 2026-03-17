import requests
import os
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from tqdm import tqdm

# Configuration
PBF_URL = "https://download.geofabrik.de/africa/nigeria-latest.osm.pbf"
SAVE_DIR = "."
CONFIG_DIR = "latestOsm"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.txt")


def download_with_resume(max_retries=5, retry_delay=5):
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    # 1. Check Metadata
    try:
        head = requests.head(PBF_URL, allow_redirects=True, timeout=10)
        head.raise_for_status()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    target_filename = os.path.basename(head.url)
    file_path = os.path.join(CONFIG_DIR, target_filename)
    remote_size = int(head.headers.get("Content-Length", 0))
    remote_time = parsedate_to_datetime(head.headers.get("Last-Modified"))

    # 2. Check if already up to date
    if os.path.exists(file_path):
        local_size = os.path.getsize(file_path)
        local_time = datetime.fromtimestamp(
            os.path.getmtime(file_path), tz=remote_time.tzinfo
        )
        if local_size == remote_size and local_time >= remote_time:
            print(f"No update needed. '{target_filename}' is current.")
            return

    # 3. Resumable Download Loop
    print(f"Starting download: {target_filename}")
    current_pos = 0

    for attempt in range(max_retries):
        try:
            # Check current local size for resume
            if os.path.exists(file_path):
                current_pos = os.path.getsize(file_path)

            if current_pos >= remote_size:
                break  # Finished

            # Request only the missing bytes
            headers = {"Range": f"bytes={current_pos}-"}
            response = requests.get(PBF_URL, headers=headers, stream=True, timeout=15)

            # 206 means Partial Content (Successful Resume)
            mode = "ab" if current_pos > 0 and response.status_code == 206 else "wb"
            if response.status_code not in [200, 206]:
                # If server doesn't support Range, restart from 0
                mode = "wb"
                current_pos = 0

            with tqdm(
                total=remote_size,
                initial=current_pos,
                unit="B",
                unit_scale=True,
                desc=target_filename,
            ) as pbar:
                with open(file_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            current_pos += len(chunk)
                            pbar.update(len(chunk))

            # Sync timestamp and config after full success
            os.utime(file_path, (datetime.now().timestamp(), remote_time.timestamp()))
            with open(CONFIG_FILE, "w") as config:
                config.write(target_filename)
            print(f"\nDownload complete: {target_filename}")
            return

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"\nAttempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    print("\nMax retries reached. Download failed.")


if __name__ == "__main__":
    download_with_resume()
