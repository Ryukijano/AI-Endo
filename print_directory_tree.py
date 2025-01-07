# filename: print_directory_tree.py

import os
from pathlib import Path

def generate_tree_structure(base_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Directory structure for: {base_path}\n")
        f.write("=" * 50 + "\n\n")

        # Process Images directory
        images_path = os.path.join(base_path, 'Images')
        if os.path.exists(images_path):
            f.write("Images/\n")
            for video_num in range(1, 21):  # Videos 1-20
                video_dir = os.path.join(images_path, f'Video{video_num}')
                if os.path.exists(video_dir):
                    # Get first and last image numbers
                    images = sorted([f for f in os.listdir(video_dir) if f.startswith('Image')])
                    if images:
                        f.write(f"├── Video{video_num}/\n")
                        f.write(f"│   ├── {images[0]}\n")
                        f.write(f"│   └── {images[-1]}\n")

        # Process Labels directory
        labels_path = os.path.join(base_path, 'Labels')
        if os.path.exists(labels_path):
            f.write("\nLabels/\n")
            label_files = sorted([f for f in os.listdir(labels_path) if f.startswith('Phase')])
            for i, label_file in enumerate(label_files):
                if i == len(label_files) - 1:
                    f.write(f"└── {label_file}\n")
                else:
                    f.write(f"├── {label_file}\n")

def main():
    exvivo_path = '/home/ryukijano/AI-Endo/DATA_ROOT_EXVIVO'
    output_file = 'tree_structure.txt'
    
    if not os.path.exists(exvivo_path):
        print(f"Error: Path {exvivo_path} does not exist")
        return
    
    print(f"Generating directory tree structure for: {exvivo_path}")
    generate_tree_structure(exvivo_path, output_file)
    print(f"Tree structure has been written to: {output_file}")

    # Print summary statistics
    total_videos = len([d for d in os.listdir(os.path.join(exvivo_path, 'Images')) 
                       if d.startswith('Video')])
    total_phase_files = len([f for f in os.listdir(os.path.join(exvivo_path, 'Labels')) 
                           if f.startswith('Phase')])
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\nSummary Statistics\n")
        f.write("=" * 20 + "\n")
        f.write(f"Total Video Directories: {total_videos}\n")
        f.write(f"Total Phase Label Files: {total_phase_files}\n")

if __name__ == "__main__":
    main()