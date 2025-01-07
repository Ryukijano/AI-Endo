import os
import pickle
from collections import defaultdict

def analyze_embeddings(emb_file, data_dict_file):
    """Analyze embeddings and compare with original data"""
    print("\n=== Embeddings Analysis ===")
    
    # Load data
    with open(emb_file, 'rb') as f:
        feature_embs = pickle.load(f)
    
    with open(data_dict_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Analyze structure
    print("\nStructure Analysis:")
    print(f"Number of videos in data_dict: {len(data_dict)}")
    print(f"Number of videos in embeddings: {len(feature_embs)}")
    
    # Compare video by video
    print("\nDetailed Analysis:")
    for video_name in sorted(data_dict.keys()):
        print(f"\n{video_name}:")
        img_count = len(data_dict[video_name]['img'])
        emb_count = len(feature_embs.get(video_name, []))
        print(f"  Images in data_dict: {img_count}")
        print(f"  Features generated: {emb_count}")
        
        if img_count != emb_count:
            print(f"  WARNING: Count mismatch!")
            
        if video_name not in feature_embs:
            print(f"  ERROR: No embeddings generated!")
        elif len(feature_embs[video_name]) > 0:
            print(f"  Feature shape: {feature_embs[video_name][0].shape}")

def main():
    data_dict_file = '/home/ryukijano/AI-Endo/DATA_DICT_EXVIVO.pkl'
    emb_file = '/home/ryukijano/AI-Endo/model/emb_ESDSafety2025-06-21-04-45.pkl'
    
    if not os.path.exists(emb_file):
        print(f"Error: Embeddings file not found: {emb_file}")
        return
        
    analyze_embeddings(emb_file, data_dict_file)

if __name__ == "__main__":
    main()