# YOLOCOCO

The COCO (Common Objects in Context) dataset of 80 object categories and over 200K labeled images is a large-scale object detection, segmentation, and captioning dataset. You can explore the classes here: https://cocodataset.org/#explore

This script was used for a project and builds upon tikitong's minicoco script (repo: https://github.com/tikitong/minicoco, solution: https://stackoverflow.com/a/73249837/14864907) and generates a training, test and validation dataset in YOLO format, with the following directory tree:

```
dataset/
  train/
    images/ *.jpg
    labels/ *.txt
  test/
    images/ *.jpg
    labels/ *.txt
  valid
    images/ *.jpg
    labels/ *.txt
```

### Create and activate a virtual environment:
```
python -m venv yolococo

# in Windows:
yolococo/Scripts/Activate.ps1

# in Linux:
source yolococo/bin/activate

pip install -r requirements.txt
```

### Run the script: 
To create a dataset of 100 training images, 10 test images and 10 validation images for the categories "fork", "knife" and "spoon", run the following:

```
python script.py -train 100 -test 10 -valid 10 -cats fork knife spoon
```
