import argparse
import sys

# pull the Ultralytics "yolov3" library first
sys.path.append("/")

from models import *
from utils.datasets import *
from utils.utils import *
from drone_command import *


def detect():
    # For the Tello drone, this should be `udp://<LOCAL IP>:111111`

    source = 'udp://192.168.10.1:11111'
    drone, drone_state, start_time = init_drone()


    # Initialize
    device = torch_utils.select_device(opt.device)

    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)

    # Load weights
    model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=opt.img_size)

    # Run inference
    t0 = time.time()

    run_inference(dataset, device, model, opt.img_size, half, drone, drone_state, start_time)

    print('Done. (%.3fs)' % (time.time() - t0))


def run_inference(dataset, device, model, imgsz, half, drone, drone_state, start_time):
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    # Once per frame
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, multi_label=False)

        bounding_boxes, im0 = process_detections(im0s, img, path, pred, t1, t2)

        if time.time() - start_time > 180:
            drone.land()
            break

        if im0 is not None and len(bounding_boxes) > 0:
            command_successful, drone_state = sendCommandToDrone(drone, bounding_boxes, im0.shape, drone_state)


def process_detections(im0s, img, path, pred, t1, t2):
    bounding_boxes = []
    im0 = None
    # Process detections
    for i, det in enumerate(pred):  # detections for image i
        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

        if det is not None and len(det):
            s += f'{det.shape[0]:g} person(s) '

            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(0, 137, 146))

            bounding_boxes.append(det[:, :4])

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        cv2.imshow(p, im0)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration
    return bounding_boxes, im0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='model/yolov3-tiny-person.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='model/data/person-coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='rocket-person-tiny.pt', help='weights path')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--droneip', type=str, default='192.168.10.1', help='drone IP :D')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
