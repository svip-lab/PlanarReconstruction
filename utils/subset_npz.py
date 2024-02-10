import shutil
import os

def main():
    # Path to the directory containing all .npz files
    npz_directory = '/Users/dimafadeev/Desktop/Catalog/TUM/WS23/ML3D/repo/processed_data/train'

    # Path to the .txt file containing the list of files to subset
    txt_file_path = '/Users/dimafadeev/Desktop/Catalog/TUM/WS23/ML3D/repo/processed_data/train.txt'

    # Path to the directory where you want to save the subset
    subset_directory = '/Users/dimafadeev/Desktop/Catalog/TUM/WS23/ML3D/repo/processed_data/train_subset'

    # Make sure the subset directory exists
    os.makedirs(subset_directory, exist_ok=True)

    # Read the list of .npz file names from the .txt file
    with open(txt_file_path, 'r') as file:
        subset_files = [line.strip() for line in file]

    # Copy the subset .npz files
    for file_name in subset_files:
        full_file_path = os.path.join(npz_directory, file_name)
        if os.path.isfile(full_file_path):
            # Copy the file to the subset directory
            shutil.copy(full_file_path, subset_directory)
        else:
            print(f"File {file_name} not found in the npz directory.")

if __name__ == "__main__":
    main()
