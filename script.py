import os
import shutil
import random
from PIL import Image
import json
import argparse
import os
import zipfile
import requests
import subprocess
import os
import json
import argparse
import numpy as np
from pathlib import Path
from random import sample
from pycocotools.coco import COCO
from alive_progress import alive_bar
import asyncio

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

parser = argparse.ArgumentParser()

parser.add_argument("-train", "--training", type=int, help="number of training images")
parser.add_argument("-test", "--test", type=int, help="number of images in the test set.")
parser.add_argument("-valid", "--validation", type=int, help="number of images in the validation set.")
parser.add_argument("-cats", "--nargs", nargs='+', help="category names.")
args = parser.parse_args()

async def download_coco():

    print("Downloading COCO annotations...")

    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    response = requests.get(url)
    with open("annotations_trainval2017.zip", "wb") as f:
        f.write(response.content)

    # Open the zip file
    with zipfile.ZipFile("annotations_trainval2017.zip", 'r') as zip_ref:
        # Extract all the contents to the specified directory
        zip_ref.extractall(".")

async def main():
    await download_coco()

asyncio.run(main())

#######################MINICOCO######################################

Path("data/images").mkdir(parents=True, exist_ok=True)
Path("data/labels").mkdir(parents=True, exist_ok=True)

coco = COCO("annotations/instances_train2017.json")
catNms = args.nargs
catIds = coco.getCatIds(catNms)
imgIds = coco.getImgIds(catIds=catIds)

imgOriginals = coco.loadImgs(imgIds)
imgShuffled = sample(imgOriginals, len(imgOriginals))

annotations = {"info": {
    "description": "my-project-name"
}
}

def myImages(images: list, train: int, val: int) -> tuple:
    myImagesTrain = images[:train]
    myImagesVal = images[train:train+val]
    return myImagesTrain, myImagesVal

def cocoJson(images: list) -> dict:
    '''getCatIds return a sorted list of id.
    for the creation of the json file in coco format, the list of ids must be successive 1, 2, 3..
    so we reorganize the ids. In the cocoJson method we modify the values of the category_id parameter.'''

    dictCOCO = {k: coco.getCatIds(k)[0] for k in catNms}
    dictCOCOSorted = dict(sorted(dictCOCO.items(), key=lambda x: x[1]))

    IdCategories = list(range(1, len(catNms)+1))
    categories = dict(zip(list(dictCOCOSorted), IdCategories))

    arrayIds = np.array([k["id"] for k in images])
    annIds = coco.getAnnIds(imgIds=arrayIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for k in anns:
        k["category_id"] = catIds.index(k["category_id"])+1
    cats = [{'id': int(value), 'name': key}
            for key, value in categories.items()]
    annotations["images"] = images
    annotations["annotations"] = anns
    annotations["categories"] = cats

    return annotations

def createJson(jsonfile: json, train: bool) -> None:
    name = "train"
    if not train:
        name = "val"
    with open(f"data/labels/{name}.json", "w") as outfile:
        json.dump(jsonfile, outfile)


def downloadImages(img: list, title: str) -> None:
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    with alive_bar(len(img), title=title) as bar:
        for im in img:
            if not os.path.isfile(f"data/images/{im['file_name']}"):
                img_data = session.get(im['coco_url']).content
                with open('data/images/' + im['file_name'], 'wb') as handler:
                    handler.write(img_data)
            bar()


imagetrain, imageval = myImages(imgShuffled, args.training, args.validation)

trainset = cocoJson(imagetrain)
createJson(trainset, train=True)
downloadImages(imagetrain, title='Downloading images of the training set:')

valset = cocoJson(imageval)
createJson(valset, train=False)
downloadImages(imageval, title='Downloading images of the validation set:')

#####################################################################

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

def coco_to_yolo(bbox, image_width, image_height):
    x_min, y_min, width, height = bbox

    # Calculate center coordinates
    center_x = (x_min + width / 2) / image_width
    center_y = (y_min + height / 2) / image_height

    # Normalize width and height
    normalized_width = width / image_width
    normalized_height = height / image_height

    return [center_x, center_y, normalized_width, normalized_height]

def create_txt_file(image_info, annotation_info, txt_filepath):

    source_filepath = os.path.join('data', 'images', image_info['file_name'])
    image_width, image_height = get_image_dimensions(source_filepath)

    with open(txt_filepath, 'w') as txt_file:
        # Write file name to the text file
        #txt_file.write(image_info['file_name'] + '\n')

        # Iterate through annotations and write class_id and bounding box info
        for annotation in annotation_info:
            class_id = annotation['category_id']
            coco_bbox = annotation['bbox']
            
            # Convert COCO bounding box to YOLO format
            yolo_bbox = coco_to_yolo(coco_bbox, image_width, image_height)

            # Write class_id and YOLO-formatted bounding box info to the text file
            txt_file.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

def create_dataset_folders(dataset_path):
    folders = ['train', 'valid']
    subfolders = ['images', 'labels']

    for folder in folders:
        for subfolder in subfolders:
            path = os.path.join(dataset_path, folder, subfolder)
            os.makedirs(path, exist_ok=True)

# Load JSON data into a variable
with open('data/labels/train.json', 'r') as train_file:
    train_data = json.load(train_file)

with open('data/labels/val.json', 'r') as val_file:
    val_data = json.load(val_file)

# Create the 'dataset' folder and its subfolders
create_dataset_folders('dataset')

# Iterate through images in the 'train' split
for image_info in train_data['images']:
    # Get corresponding annotations for the current image
    annotation_info = [annotation for annotation in train_data['annotations'] if annotation['image_id'] == image_info['id']]

    # Create a text file with the file name and write class_id and bounding box info
    txt_filename = image_info['file_name'].replace('.jpg', '.txt')
    txt_filepath = os.path.join('dataset', 'train', 'labels', txt_filename)
    create_txt_file(image_info, annotation_info, txt_filepath)

    # Copy the image to the 'images' folder
    source_filepath = os.path.join('data', 'images', image_info['file_name'])
    image_filepath = os.path.join('dataset', 'train', 'images', image_info['file_name'])
    shutil.copy(source_filepath, image_filepath)

# Iterate through images in the 'valid' split
for image_info in val_data['images']:
    # Get corresponding annotations for the current image
    annotation_info = [annotation for annotation in val_data['annotations'] if annotation['image_id'] == image_info['id']]

    # Create a text file with the file name and write class_id and bounding box info
    txt_filename = image_info['file_name'].replace('.jpg', '.txt')
    txt_filepath = os.path.join('dataset', 'valid', 'labels', txt_filename)
    create_txt_file(image_info, annotation_info, txt_filepath)

    # Copy the image to the 'images' folder
    source_filepath = os.path.join('data', 'images', image_info['file_name'])
    image_filepath = os.path.join('dataset', 'valid', 'images', image_info['file_name'])
    shutil.copy(source_filepath, image_filepath)

# Define the directories
valid_images_dir = 'dataset/valid/images'
valid_labels_dir = 'dataset/valid/labels'
test_images_dir = 'dataset/test/images'
test_labels_dir = 'dataset/test/labels'

# Create the test directories if they don't exist
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Get the list of images in the valid directory
valid_images = os.listdir(valid_images_dir)
num_valid_images = len(valid_images)

# Calculate the number of images to move to the test directory
num_test_images = num_valid_images // 2

# Randomly select images to move
test_images = random.sample(valid_images, num_test_images)

# Move the selected images and their corresponding labels to the test directory
for image in test_images:
    # Move image
    src_image_path = os.path.join(valid_images_dir, image)
    dest_image_path = os.path.join(test_images_dir, image)
    shutil.move(src_image_path, dest_image_path)

    # Move label
    label_file = image.replace('.jpg', '.txt')
    src_label_path = os.path.join(valid_labels_dir, label_file)
    dest_label_path = os.path.join(test_labels_dir, label_file)
    shutil.move(src_label_path, dest_label_path)