import os
import cv2
import shutil
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split

import re

def extract_number_from_string(text):
    match = re.search(r'(\d+)',text)
    if match:
        return int(match.group(0))
    else:
        return 0
    
def parse_xml(dataset_path,labels_dict,xml_files):
    for filename in sorted(xml_files, key=extract_number_from_string):
    # Parse the XML file
        info = xet.parse(filename)
        root = info.getroot()
    
    # Find the 'object' element in the XML and extract bounding box information
        member_object = root.find('object')
        labels_info = member_object.find('bndbox')
        xmin = int(labels_info.find('xmin').text)
        xmax = int(labels_info.find('xmax').text)
        ymin = int(labels_info.find('ymin').text)
        ymax = int(labels_info.find('ymax').text)
    
    # Get the image filename and construct the full path to the image
        img_name = root.find('filename').text
        img_path = os.path.join(dataset_path, 'images', img_name)

        # Append the extracted information to the respective lists in the dictionary
        labels_dict['img_path'].append(img_path)
        labels_dict['xmin'].append(xmin)
        labels_dict['xmax'].append(xmax)
        labels_dict['ymin'].append(ymin)
        labels_dict['ymax'].append(ymax)
    
        # Read the image to get its dimensions
        height, width, _ = cv2.imread(img_path).shape
        labels_dict['img_w'].append(width)
        labels_dict['img_h'].append(height)
    
def make_dataframe(dataset_path):
    labels_dict = dict(img_path=[], 
                       xmin=[], xmax=[], 
                       ymin=[], ymax=[], 
                       img_w=[],img_h=[]
                       )
    xml_files = glob(f'{dataset_path}/annotations/*.xml')
    
    parse_xml(dataset_path,labels_dict,xml_files)
    
    all_data = pd.DataFrame(labels_dict)
    return all_data

if __name__ == "__main__":
    data = make_dataframe("/Users/mukulagarwal/Desktop/Projects/Car_Licence_Plate_Detection/archive")
    print(data)
