import os
import pandas as pd

def generate_phase_labels():
    # Define base directory
    base_dir = '/home/ryukijano/AI-Endo'
    labels_dir = os.path.join(base_dir, 'DATA_ROOT_EXVIVO', 'Labels')
    
    # Define phase mapping
    phase_dict = {
        'idle': 0,
        'marking': 1,
        'injection': 2,
        'dissection': 3
    }
    
    # Process each label file
    for video_num in range(1, 21):  # Assuming you have Video1 through Video20
        input_file = os.path.join(labels_dir, f'Video{video_num}.txt')
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found")
            continue
            
        try:
            # Read and process the label file
            df = pd.read_csv(input_file, 
                           header=None, 
                           sep='\t',
                           names=['Frame', 'Phase'])
            
            # Verify phase labels
            df['Phase'] = df['Phase'].str.lower()
            invalid_phases = df['Phase'][~df['Phase'].isin(phase_dict.keys())]
            if not invalid_phases.empty:
                print(f"Warning: Invalid phases found in {input_file}: {invalid_phases.unique()}")
                continue
                
            # Convert phases to numeric values
            df['Phase'] = df['Phase'].map(phase_dict)
            
            # Save processed file
            output_file = os.path.join(labels_dir, f'Phase{video_num}.txt')
            df.to_csv(output_file, sep='\t', header=False, index=False)
            print(f"Processed {input_file} -> {output_file}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            
    print("Phase label generation complete")

if __name__ == "__main__":
    generate_phase_labels()