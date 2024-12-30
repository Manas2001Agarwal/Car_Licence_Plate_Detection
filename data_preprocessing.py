import shutil
import os
from sklearn.model_selection import train_test_split
from dataset import make_dataframe

def make_split_folder_in_yolo_format(split_name, split_df):
    """
    Creates a folder structure for a dataset split (train/val/test) in YOLO format.

    Parameters:
    split_name (str): The name of the split (e.g., 'train', 'val', 'test').
    split_df (pd.DataFrame): The DataFrame containing the data for the split.

    The function will create 'labels' and 'images' subdirectories under 'datasets/cars_license_plate/{split_name}',
    and save the corresponding labels and images in YOLO format.
    """
    labels_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join('datasets', 'cars_license_plate_new', split_name, 'images')

    # Create directories for labels and images
    os.makedirs(labels_path)
    os.makedirs(images_path)
    
    # Iterate over each row in the DataFrame
    for _, row in split_df.iterrows():
        img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))
        
        # Calculate YOLO format bounding box coordinates
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        # Save the label in YOLO format
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
            
        # Copy the image to the images directory
        shutil.copy(row['img_path'], os.path.join(images_path, img_name + img_extension))
    
    print(f"Created '{images_path}' and '{labels_path}'")
    # Define the content of the datasets.yaml file
    datasets_yaml = '''
        path: cars_license_plate_new

        train: train/images
        val: val/images
        test: test/images

        # number of classes
        nc: 1

        # class names
        names: ['license_plate']
    '''

    # Write the content to the datasets.yaml file
    with open('datasets/datasets.yaml', 'w') as file:
        file.write(datasets_yaml)

if __name__ == "__main__":
    alldata = make_dataframe("/Users/mukulagarwal/Desktop/Projects/Car_Licence_Plate_Detection/archive")

    train, test = train_test_split(alldata, test_size=1/10, random_state=42)
    train, val = train_test_split(train, train_size=8/9, random_state=42)
    
    make_split_folder_in_yolo_format("train",train)
    make_split_folder_in_yolo_format("val",val)
    make_split_folder_in_yolo_format("test",test)
