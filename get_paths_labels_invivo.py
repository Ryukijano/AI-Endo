import os
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
from os.path import join as pjoin

def process_data(data_name, img_dir, label_dir, phase_dict):
    """Process a single video's data"""
    img_file_dir = pjoin(img_dir, data_name)
    img_files = glob(pjoin(img_file_dir, "*.png"))
    img_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

    label_file = pjoin(label_dir, f'Phase{data_name.replace("Video", "")}.txt')
    if os.path.isfile(label_file):
        # Read the label file
        phase_label = pd.read_csv(label_file, header=None, sep="[ ]{1,}|\t", engine="python")
        if len(phase_label.columns) == 5:
            phase_label.columns = ["Frame", "Phase", "#1", "#2", "#3"]
        elif len(phase_label.columns) == 2:
            phase_label.columns = ["Frame", "Phase"]
        else:
            raise ValueError("The header of label file cannot be matched")
        
        # Convert types and replace phase labels
        phase_label = phase_label.astype({"Frame": int, "Phase": str})
        phase_label = phase_label.replace({"Phase": phase_dict}).infer_objects(copy=False)
        phase_labels = phase_label["Phase"].tolist()
        
        # Handle mismatches between images and labels
        if len(img_files) != len(phase_labels):
            print(f"Warning: Mismatch in {data_name} - {len(img_files)} images vs {len(phase_labels)} labels")
            # Trim to shorter length to ensure alignment
            min_len = min(len(img_files), len(phase_labels))
            img_files = img_files[:min_len]
            phase_labels = phase_labels[:min_len]
    else:
        print(f"Label file not found: {label_file}")
        phase_labels = None

    return data_name, {"img": img_files, "phase": phase_labels}

def main():
    root_dir = '/home/ryukijano/AI-Endo/DATA_ROOT_INVIVO'
    save_dir = "/home/ryukijano/AI-Endo"
    img_dir = os.path.join(root_dir, 'Images')
    label_dir = os.path.join(root_dir, 'Labels')

    phase_dict = {}
    phase_dict_key = ['idle', 'marking', 'injection', 'dissection']
    for i in range(len(phase_dict_key)):
        phase_dict[phase_dict_key[i]] = i

    data_names = [os.path.basename(x) for x in glob(pjoin(img_dir, "*"))]
    data_names = sorted(data_names)

    sorted_datas = {}
    for data_name in tqdm(data_names, desc="Processing InVivo dataset"):
        data_name, data_dict = process_data(data_name, img_dir, label_dir, phase_dict)
        sorted_datas[data_name] = data_dict

    save_file = os.path.join(save_dir, "DATA_DICT_INVIVO.pkl")
    print(f"Saving to {save_file}")
    with open(save_file, 'wb') as f:
        pickle.dump(sorted_datas, f)
    print(f"{len(sorted_datas)} videos saved to {save_file}")

    # Verification step
    for data_name, data_dict in sorted_datas.items():
        num_images = len(data_dict['img'])
        num_labels = len(data_dict['phase']) if data_dict['phase'] else 0
        print(f"{data_name}: {num_images} images, {num_labels} labels")
        if num_images != num_labels:
            print(f"  -> MISMATCH: {num_images} images vs {num_labels} labels")

if __name__ == "__main__":
    main()