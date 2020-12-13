# Drone footage human tracking and face detection (P45)

Kert, Agnes, Kaspar, Oliver

Course project for Machine Learning MTAT.03.227.

## Human detection

For detecting people from a video stream an object detection neural network is required. Our
solution relies on the [YOLO family of object detection models](https://pjreddie.com/darknet/yolo/).

### Setting up the training environment

1. Clone this repository:  
   `git clone https://github.com/TaliKert/drone-video-ml.git`

2. Set the repository root directory as current directory.

3. Get PyTorch with the most suitable CUDA version from here:
   https://pytorch.org/

4. Install the prerequisites of this project:
   `pip install -r requirements.txt`

5. All following instructions require the following repository be cloned to the repository root:  
   `git clone -b archive https://github.com/ultralytics/yolov3.git`  
   The repository contains necessary logic to train and use YOLO models for detection.  
   Run `pip install -r ./yolov3/requirements.txt` to install all its prerequisites.

6. To train a model which would only detect humans, a dataset is required.  
   We have utilized the Coco 2017 dataset for this purpose.  
   Download the images and labels using the script in the cloned repository:

```
$ yolov3/data/get_coco2017.sh
```

7. The extraction script can then separate all non-person images and labels from the dataset. Run it
   like so:

```
$ scripts/extract-person-coco.sh
```

8. We train on top of the pretrained object detection model to get faster results.  
   First, the model description for single class classification is in the `model` directory. We use
   the _tiny_ variant of YOLOv3, as it runs detection the fastest on limited hardware.  
   To get the pre-trained weights, you can get them
   from [the YOLO website](https://pjreddie.com/darknet/yolo/) and convert them into PyTorch weights
   with

```
$ cd yolov3
$ python -c "from models import *; convert('../model/yolov3-tiny-person.cfg', 'weights/yolov3-tiny.weights')"
```

9. Next, the model can be trained by running

```
$ cd yolov3
$ python train.py --data model/data/person-coco.data --cfg ../model/yolov3-tiny-person.cfg --weights=weights/yolov3-tiny.pt --single-cls --batch-size 32 --epochs 20
```

## Drone control

We used easyTello Python library: https://github.com/Virodroid/easyTello

### Running the drone control

(The first 4 following steps are almost the same as the first 5 steps of setting up the training
environment)

1. Clone this repository:  
   `git clone https://github.com/TaliKert/drone-video-ml.git`

2. Set the repository root directory as current directory.

3. Get PyTorch with the most suitable CUDA version from here:
   https://pytorch.org/

4. All following instructions require the following repository be cloned to the repository root:  
   `git clone -b archive https://github.com/ultralytics/yolov3.git`  
   The repository contains necessary logic to train and use YOLO models for detection.  
   Run `pip install -r ./yolov3/requirements.txt` to install all its prerequisites.

5. Get our weights from here:
   https://drive.google.com/drive/folders/1f6UPxNfCGXa8vIfpTN5_EnIkbZSiDs6l  
   These places are to be placed to the repository root directory.

6. Make sure you are connected to the drone wifi.

7. Use `python drone.py` to run the drone control script.