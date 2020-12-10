import argparse
import sys

# pull the Ultralytics "yolov3" library first
sys.path.append("yolov3/")

from models import *
from utils.datasets import *
from utils.utils import *
from drone_command import *


def detect():
    imgsz = opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    # ip = opt.droneip
    # For the Tello drone, this should be `udp://<LOCAL IP>:111111`
    source = 'rtsp://' + "192.168.1.232" + ':5554/camera'

    # Initialize
    device = torch_utils.select_device(opt.device)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz)

    # Run inference
    t0 = time.time()

    run_inference(dataset, device, model, imgsz, half)

    print('Done. (%.3fs)' % (time.time() - t0))


def run_inference(dataset, device, model, imgsz, half):
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
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        bounding_boxes, im0 = process_detections(im0s, img, path, pred, t1, t2)

        if im0 is not None and len(bounding_boxes) > 0:
            sendCommandToDrone(bounding_boxes, im0.shape)


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
                plot_one_box(xyxy, im0, label=label, color=(0, 255, 255))

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
    parser.add_argument('--cfg', type=str, default='model/yolov3-tiny-person', help='*.cfg path')
    parser.add_argument('--names', type=str, default='model/data/person-coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='rocket-preson-tiny.pt', help='weights path')
    parser.add_argument('--source', type=str, default='192.168.0.101', help='source')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--droneip', action='store_true', help='drone IP :D')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
