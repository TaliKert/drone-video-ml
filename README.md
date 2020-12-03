# Drone footage human tracking and face detection (P45)

Kert, Agnes, Kaspar, Oliver

Course project for Machine Learning MTAT.03.227.

## Human tracking

For detecting people from a video stream an object detection neural network is required.
Our solution relies on the [YOLO family of object detection models](https://pjreddie.com/darknet/yolo/).

All following instructions require the following repository be cloned to this repository's root directory:

`git clone -b archive https://github.com/ultralytics/yolov3.git`

The repository contains necessary logic to train and use YOLO models for detection.
Run `pip install -r requirements.txt` to install all its prerequisites.

### Setting up the training environment

To train a model which would only detect humans, a dataset is required.
We have utilized the Coco 2017 dataset for this purpose.

Download the images and labels using the script in the cloned repository:
```
$ yolov3/data/get_coco2017.sh
```

The extraction script can then separate all non-person images and labels from the dataset.
Run it like so:
```
$ scripts/extract-person-coco.sh
```

We utilized transfer learning on a pretrained object detection model to get faster results.

First, the model description for single class classification is in the `model` directory.
We use the _tiny_ variant of YOLOv3, as it runs detection the fastest on limited hardware.

To get the pre-trained weights, you can get them from [the YOLO website](https://pjreddie.com/darknet/yolo/) and convert them into Pytorch weights with
```
$ cd yolov3
$ python -c "from models import *; convert('../model/yolov3-tiny-person.cfg', 'weights/yolov3-tiny.weights')"
```

Next, the model can be trained by running
```
$ cd yolov3
$ python train.py --data model/data/person-coco.data --cfg ../model/yolov3-tiny-person.cfg --weights=weights/yolov3-tiny.pt --single-cls --batch-size 32 --epochs 20
```
