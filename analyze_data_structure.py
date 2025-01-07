import os
import pickle
from glob import glob
from collections import defaultdict

def analyze_image_files(root_dir):
    """Analyze image files in the data root directory"""
    print("\n=== Image Files Analysis ===")
    
    image_dir = os.path.join(root_dir, 'Images')
    video_dirs = sorted(glob(os.path.join(image_dir, 'Video*')))
    
    file_stats = defaultdict(dict)
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        image_files = sorted(glob(os.path.join(video_dir, 'Image*.png')))
        
        if image_files:
            first_img = os.path.basename(image_files[0])
            last_img = os.path.basename(image_files[-1])
            file_stats[video_name] = {
                'count': len(image_files),
                'first': first_img,
                'last': last_img,
                'range': f"{first_img} to {last_img}"
            }
    
    print("\nImage file statistics:")
    for video, stats in file_stats.items():
        print(f"\n{video}:")
        print(f"  Count: {stats['count']} images")
        print(f"  Range: {stats['range']}")
    
    return file_stats

def analyze_label_files(root_dir):
    """Analyze label files in the data root directory"""
    print("\n=== Label Files Analysis ===")
    
    label_dir = os.path.join(root_dir, 'Labels')
    phase_files = sorted(glob(os.path.join(label_dir, 'Phase*.txt')))
    
    label_stats = {}
    for phase_file in phase_files:
        video_num = os.path.basename(phase_file).replace('Phase', '').replace('.txt', '')
        video_name = f"Video{video_num}"
        
        with open(phase_file, 'r') as f:
            lines = f.readlines()
            first_line = lines[0].strip()
            last_line = lines[-1].strip()
            
        label_stats[video_name] = {
            'count': len(lines),
            'first': first_line,
            'last': last_line
        }
    
    print("\nLabel file statistics:")
    for video, stats in label_stats.items():
        print(f"\n{video}:")
        print(f"  Count: {stats['count']} labels")
        print(f"  Range: {stats['first']} to {stats['last']}")
    
    return label_stats

def analyze_pickle_files():
    """Analyze the pickle files containing data dictionaries"""
    print("\n=== Pickle Files Analysis ===")
    
    pickle_files = {
        'ExVivo': '/home/ryukijano/AI-Endo/DATA_DICT_EXVIVO.pkl',
        'InVivo': '/home/ryukijano/AI-Endo/DATA_DICT_INVIVO.pkl'
    }
    
    for dataset_type, pickle_file in pickle_files.items():
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                data_dict = pickle.load(f)
            
            print(f"\n{dataset_type} Dataset:")
            for video_name, video_data in sorted(data_dict.items()):
                print(f"\n{video_name}:")
                print(f"  Images: {len(video_data['img'])}")
                if video_data.get('phase') is not None:
                    print(f"  Labels: {len(video_data['phase'])}")
                else:
                    print("  Labels: None")

def main():
    root_dir = '/home/ryukijano/AI-Endo/DATA_ROOT_EXVIVO'
    
    print("Analyzing data structure...")
    file_stats = analyze_image_files(root_dir)
    label_stats = analyze_label_files(root_dir)
    analyze_pickle_files()
    
    # Compare image and label counts
    print("\n=== Consistency Check ===")
    for video_name in file_stats.keys():
        if video_name in label_stats:
            img_count = file_stats[video_name]['count']
            label_count = label_stats[video_name]['count']
            if img_count != label_count:
                print(f"\nMismatch in {video_name}:")
                print(f"  Images: {img_count}")
                print(f"  Labels: {label_count}")

if __name__ == "__main__":
    main()