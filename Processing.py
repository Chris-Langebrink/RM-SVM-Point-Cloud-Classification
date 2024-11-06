import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def filter():
    directory = os.getcwd()
    out_directory = 'filtered training'

    # Get all the .labels and .txt files
    label_files = [f for f in os.listdir(directory) if f.endswith('.labels')]
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # Iterate through each pair of label and txt files
    for label_file in label_files:
        # Find the corresponding .txt file
        corresponding_txt_file = label_file.replace('.labels', '.txt')
        
        # Check if the corresponding txt file exists
        if corresponding_txt_file in txt_files:
            # Open the label and corresponding point cloud files
            with open(os.path.join(directory, label_file), 'r') as labels, \
                open(os.path.join(directory, corresponding_txt_file), 'r') as points:
                # Create a new file to write the updated label data
                updated_label_file = os.path.join(out_directory, label_file.replace('.labels', '_filtered.txt'))
                with open(updated_label_file, 'w') as output_file:
                    # Loop through both files line by line
                    for label, point in zip(labels, points):
                        # Check if the label is not '0'
                        if label.strip() != '0':
                            # Append the corresponding point data to the label
                            updated_label = label.strip() + " " + point.strip()
                            # Write the updated label to the output file
                            output_file.write(updated_label + "\n")
                            
            print(f"Updated labels have been written to '{updated_label_file}'.")

    print("All files have been processed.")

def subsample():
    directory = 'filtered training cloud'
    all_filtered_data = pd.DataFrame()


    # Loop through all .txt files in the directory
    for filename in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        
        # Load the data into a DataFrame
        df = pd.read_csv(filepath, skiprows=1, header=None)
        df.dropna(inplace=True)
        
        # Re-reading the file to get the headers
        with open(filepath, 'r') as file:
            headers = file.readline().strip().split(',')
        
        df.columns = headers  # Set the column headers

        # Column index for Scalar field
        scalar_field_col = 'Scalar field'

        # Filter rows for each scalar value, limit to 20,000 rows
        for scalar_value in range(1, 9):
            scalar_rows = df[df[scalar_field_col] == scalar_value]
            scalar_rows = scalar_rows.head(1000)
            all_filtered_data = pd.concat([all_filtered_data, scalar_rows])


    output_filepath = 'combined_training_data_small.txt'
    all_filtered_data.to_csv(output_filepath, index=False, header=True)

    print("Processing complete.")


def plotHist():
    file1_path = 'combined_training_data2.txt'
    file2_path = 'bildstein_station5_xyz_intensity_rgb_filtered - Cloud.txt'

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    scalar_field_col = 'Scalar field'

    # Count the number of rows for each scalar value
    scalar_counts_1 = df1[scalar_field_col].value_counts().sort_index()
    scalar_counts_2 = df2[scalar_field_col].value_counts().sort_index()

    # Define the labels for the x-axis
    labels = [
    'man-made terrain', 'natural terrain', 'high vegetation', 
    'low vegetation', 'buildings', 'hard scape', 'scanning artefacts', 'cars'
    ]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, scalar_counts_1.values, width, label='Filtered Combined Data')
    rects2 = ax.bar(x + width/2, scalar_counts_2.values, width, label='bildstein_station1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Label')
    ax.set_ylabel('Number of Points')
    ax.set_title('Comparison of Scalar Value Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    # Add value labels on top of the bars
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # subsample()
    plotHist()