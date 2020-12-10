import argparse
import sys
# pull the Ultralytics "yolov3" library first
sys.path.append("yolov3/")

from models import *
from utils.datasets import *
from utils.utils import *


def detect():
    pass


if __name__ == '__main__':
    with torch.no_grad():
        detect()
