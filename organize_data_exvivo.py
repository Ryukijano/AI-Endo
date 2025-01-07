import os
import shutil
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def copy_images(case_path, video_dir):
    """Copy images while maintaining order and renaming to standard format"""
    case_name = os.path.basename(case_path)
    image_files = sorted(glob(os.path.join(case_path, f'{case_name}-*.png')))
    
    for idx, img_file in enumerate(image_files, 1):
        dst = os.path.join(video_dir, f'Image{str(idx).zfill(5)}.png')
        shutil.copy2(img_file, dst)

def process_case(case_path, idx, exvivo_root, base_dir):
    case_name = os.path.basename(case_path)
    
    # Create video directory
    video_dir = os.path.join(exvivo_root, 'Images', f'Video{idx}')
    os.makedirs(video_dir, exist_ok=True)
    
    # Copy images
    copy_images(case_path, video_dir)
    
    # Copy and rename annotation file
    src_label = os.path.join(base_dir, 'ExvivoTrialAnnotation', f'{case_name}.txt')
    if os.path.exists(src_label):
        dst_label = os.path.join(exvivo_root, 'Labels', f'Phase{idx}.txt')
        shutil.copy2(src_label, dst_label)
        print(f"Copied label file: {src_label} -> {dst_label}")
    else:
        print(f"Warning: No label file found for {case_name}")

def create_exvivo_structure():
    # Define base directories
    base_dir = '/home/ryukijano/AI-Endo'
    exvivo_root = os.path.join(base_dir, 'DATA_ROOT_EXVIVO')
    
    # Create directory structure
    os.makedirs(os.path.join(exvivo_root, 'Images'), exist_ok=True)
    os.makedirs(os.path.join(exvivo_root, 'Labels'), exist_ok=True)

    # Get all cases (A, B, C, D)
    case_paths = sorted(glob(os.path.join(base_dir, 'ExvivoAnimalTrial/Case_*')))
    
    print("Processing Exvivo cases...")
    with ProcessPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(process_case, case_paths, range(1, len(case_paths) + 1), [exvivo_root] * len(case_paths), [base_dir] * len(case_paths)), total=len(case_paths)))

if __name__ == "__main__":
    create_exvivo_structure()