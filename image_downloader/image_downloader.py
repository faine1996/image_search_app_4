import os
import aiohttp
import asyncio
import logging
import json

# Setup Logging
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_folder, "downloader.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
IMAGE_URLS_FILE = os.path.join(BASE_DIR, "image_urls.txt")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "images")
URL_MAP_FILE = "image_url_map.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output folder exists

# Global dictionary to map image filenames to URLs
image_url_map = {}

async def download_image(session, url, index):
    """
    Asynchronously downloads a single image and saves it to the OUTPUT_FOLDER.

    Args:
        session: aiohttp session to reuse connections.
        url: Image URL.
        index: Unique index for file naming.
    """
    file_name = os.path.join(OUTPUT_FOLDER, os.path.basename(url))
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.read()
                with open(file_name, "wb") as file:
                    file.write(content)  # Save image content to file
                logging.info(f"SUCCESS: {url} -> {file_name}")
                print(f"Downloaded: {url} - saved to {file_name}")
                image_url_map[f"image_{index}.npy"] = url  # Map filename to URL
                return True
            else:
                logging.error(f"FAILED: {url} - HTTP {response.status}")
                print(f"Failed (HTTP {response.status}): {url}")
    except Exception as e:
        logging.error(f"ERROR: {url} - {str(e)}")
        print(f"Error: {url} - {str(e)}")
    return False

async def main():
    """
    Reads URLs from a file and downloads images concurrently using aiohttp and asyncio.
    """
    # Step 1: Load URLs from file
    print("Current Working Directory:", os.getcwd())
    print("Looking for:", IMAGE_URLS_FILE)
    with open(IMAGE_URLS_FILE, "r") as file:
        urls = [line.strip() for line in file if line.strip()]
    print(f"Found {len(urls)} URLs to download.")

    # Step 2: Download images asynchronously
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url, index) for index, url in enumerate(urls)]
        results = await asyncio.gather(*tasks)  # Run all tasks concurrently

    # Step 3: Save URL map to file
    with open(URL_MAP_FILE, "w") as file:
        json.dump(image_url_map, file, indent=4)
    print(f"URL map saved to {URL_MAP_FILE}")
    logging.info(f"URL map saved to {URL_MAP_FILE}")

    # Step 4: Report results
    success_count = sum(results)
    print(f"\nDownload complete: {success_count}/{len(urls)} images downloaded successfully.")
    logging.info(f"SUMMARY: {success_count}/{len(urls)} images downloaded successfully.")

    #Signalize its done
    open(f'{OUTPUT_FOLDER}/completion_marker', 'w').close()

if __name__ == "__main__":
    # Run the main coroutine
    asyncio.run(main())
