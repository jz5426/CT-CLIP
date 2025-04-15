import os
import shutil
import random

def sample_and_migrate_files(input_dir, output_dir, sample_ratio=0.2):
    # Get list of all file paths (recursive)
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    
    # Sample 20% of the files
    num_to_sample = int(len(all_files) * sample_ratio)
    sampled_files = random.sample(all_files, num_to_sample)

    # Copy sampled files to output_dir with original structure
    for src_path in sampled_files:
        # Determine relative path
        rel_path = os.path.relpath(src_path, input_dir)
        dst_path = os.path.join(output_dir, rel_path)
        
        # Ensure output subdirectory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dst_path)
    
    print(f"Sampled {len(sampled_files)} files out of {len(all_files)}.")
    print(f"Copied to: {output_dir}")

# Example usage
input_directory = '/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16'
output_directory = '/cluster/projects/mcintoshgroup/publicData/CT-RATE-Processed/benchmark/CTRATE_Volumes_raw_h5_fp16_val'
sample_and_migrate_files(input_directory, output_directory)
