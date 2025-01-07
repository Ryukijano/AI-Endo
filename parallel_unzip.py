import os
import zipfile
from multiprocessing import Pool
from tqdm import tqdm
import logging

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        filename='parallel_unzip.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def categorize_zip(zip_filename):
    """
    Determine the category of the zip file based on its name.
    Modify this function based on your zip file naming conventions.
    For example, if label zips contain 'Label' in their name.
    """
    if 'label' in zip_filename.lower():
        return 'Labels'
    else:
        return 'Images'

def unzip_file(args):
    """
    Unzip a single zip file into the appropriate output directory.

    Parameters:
    - args (tuple): Contains zip_path and output_base_dir

    Returns:
    - tuple: (zip_path, status message)
    """
    zip_path, output_base_dir = args
    zip_filename = os.path.basename(zip_path)
    category = categorize_zip(zip_filename)
    extract_dir_name = os.path.splitext(zip_filename)[0]
    extract_path = os.path.join(output_base_dir, category, extract_dir_name)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            os.makedirs(extract_path, exist_ok=True)
            zip_ref.extractall(extract_path)
        logging.info(f"Successfully extracted: {zip_filename} to {extract_path}")
        return (zip_path, "Success")
    except zipfile.BadZipFile:
        logging.error(f"Bad zip file: {zip_filename}")
        return (zip_path, "Failed: Bad Zip File")
    except Exception as e:
        logging.error(f"Error extracting {zip_filename}: {e}")
        return (zip_path, f"Failed: {e}")

def main():
    # Setup logging
    setup_logging()

    # Hardcoded Paths
    input_dir = '/home/ryukijano/AI-Endo/23506866'
    output_dir = '/home/ryukijano/AI-Endo/23506866_extracted'
    n_jobs = 24  # Number of parallel processes

    # Validate input directory
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return

    # Create output subdirectories if they don't exist
    images_dir = os.path.join(output_dir, 'Images')
    labels_dir = os.path.join(output_dir, 'Labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Gather all zip files in the input directory
    zip_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith('.zip') and os.path.isfile(os.path.join(input_dir, f))
    ]

    if not zip_files:
        logging.warning(f"No zip files found in the specified directory: {input_dir}")
        return

    logging.info(f"Found {len(zip_files)} zip files in {input_dir}. Starting extraction...")

    # Prepare arguments for multiprocessing
    pool_args = [(zip_path, output_dir) for zip_path in zip_files]

    # Use a pool of workers to unzip files in parallel
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(unzip_file, pool_args), total=len(pool_args)))

    # Summarize results
    success = [f for f, status in results if status == "Success"]
    failed = [(f, status) for f, status in results if status != "Success"]

    logging.info(f"\nUnzipping completed. Success: {len(success)}, Failed: {len(failed)}")

    if failed:
        logging.warning("\nFailed to unzip the following files:")
        for f, status in failed:
            logging.warning(f"{os.path.basename(f)}: {status}")

if __name__ == "__main__":
    main()