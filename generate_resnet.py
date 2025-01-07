import os
import shutil
import warnings
warnings.filterwarnings("ignore")
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.esd import ESDDataset
from utils.parser import ParserUse

from model.resnet import ResNet


def generate_features(args):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize and load the ResNet model
    model = ResNet(out_channels=args.out_classes, has_fc=False)
    paras = torch.load(args.resnet_model)["model"]
    paras = {k: v for k, v in paras.items() if "fc" not in k}
    paras = {k: v for k, v in paras.items() if "embed" not in k}
    model.load_state_dict(paras, strict=True)
    model.cuda()
    model.eval()

    # Load the data dictionary
    with open(args.data_file, "rb") as f:
        data_dict = pickle.load(f)

    # Create the dataset and dataloader
    emb_dataset = ESDDataset(
        data_dict=data_dict,
        data_idxs=args.train_names + args.val_names + args.test_names,
        is_train=False,
        get_name=True,
        has_label=args.has_label
    )
    print("Data Indexes (data_idxs):", emb_dataset.data_idxs)
    emb_loader = DataLoader(
        dataset=emb_dataset,
        batch_size=args.resnet_train_bs,
        num_workers=args.num_worker,
        shuffle=False,
        drop_last=False
    )

    feature_embs = {}
    with torch.no_grad():
        for data in tqdm(emb_loader, total=len(emb_loader)):
            imgs, video_names = data  # data is a tuple (imgs, video_names)
            print("Batch Video Names:", video_names)  # Debugging
            imgs = imgs.cuda(non_blocking=True)
            video_names = list(video_names)  # Convert list of strings

            # Generate features using the model
            img_features = model(imgs).cpu().numpy()

            # Aggregate features per video
            for idx, video_name in enumerate(video_names):
                if video_name in feature_embs:
                    feature_embs[video_name].append(img_features[idx])
                else:
                    feature_embs[video_name] = [img_features[idx]]

    # Check if the number of features matches the number of images
    with open(args.data_file, "rb") as f:
        all_data = pickle.load(f)

    data_names = list(feature_embs.keys())
    for data_name in data_names:
        if data_name not in all_data:
            print(f"Warning: {data_name} is not present in data_dict.")
            continue
        if len(all_data[data_name]["img"]) != len(feature_embs[data_name]):
            print(f"Error in data {data_name}")
            print(f"Number of images {len(all_data[data_name]['img'])}")
            print(f"Number of features {len(feature_embs[data_name])}")
            raise ValueError("#imgs != #features")

    # Save the aggregated features
    args.emb_file = os.path.join(os.path.dirname(args.emb_file), f"emb_ESDSafety{args.log_time}.pkl")
    print(">>>" * 10, "Emb dataset saved to ", args.emb_file)
    with open(args.emb_file, "wb") as f:
        pickle.dump(feature_embs, f)

    # Optionally, save features as *.npy files (commented out)
    # if os.path.isdir(args.features_folder):
    #     shutil.rmtree(args.features_folder)
    # os.makedirs(args.features_folder)
    # for data_name, features in feature_embs.items():
    #     features = np.stack(features, axis=0)
    #     save_file = os.path.join(args.features_folder, f"{data_name}.npy")
    #     with open(save_file, "wb") as f:
    #         np.save(f, features)

    return args

def __getitem__(self, idx):
    # Get image path - structure: "Images/Video1/Image00001.png"
    img_path = self.img_files[idx]
    
    # Get video name from path
    video_name = os.path.basename(os.path.dirname(img_path))  # extract "Video1" from path
    
    # Load and process image
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    
    if self.is_train:
        img = self.augs(img)
    else:
        img = self.test_augs(img)
    
    if self.has_label:
        phase = self.phase_labels[idx]
        if self.get_name:
            return img, video_name  # Return video_name instead of just image name
        return img, phase
    else:
        if self.get_name:
            return img, video_name  # Return video_name instead of just image name
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='train', required=True, type=str,
                        help='Your detailed configuration of the network')

    args = parser.parse_args()
    args = ParserUse(args.cfg, log="generate").add_args(args)

    ckpts = args.makedir()
    generate_features(args)