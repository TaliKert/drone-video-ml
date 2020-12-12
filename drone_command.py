import time

import cv2
from easytello import tello

from enum import Enum

ERROR_BOUNDARY = 50


class ControlStates(Enum):
    ON_GROUND = 1
    CENTER = 2
    LAND = 3


def init_drone():
    drone = tello.Tello()
    drone.takeoff()
    send_rc(drone, 0, 0, 0, 0)
    drone.up(50)
    drone.send_command('streamon')
    return drone, ControlStates.CENTER, time.time()


def send_rc(drone, a, b, c, d):
    command = 'rc {} {} {} {}'.format(a, b, c, d)
    print('Sending command: {}'.format(command))
    drone.socket.sendto(command.encode('utf-8'), drone.tello_address)


# P - controller, maybe should try PD-controller
def proportional_centering(drone, bounding_box_center, frame_center):
    KP = 0.1
    error = frame_center - bounding_box_center
    """
    Send RC control via four channels.
        a: left/right (-100~100)
        b: forward/backward (-100~100)
        c: up/down (-100~100)
        d: yaw (-100~100)
    """
    return send_rc, (drone, 0, 0, 0, -int(KP * error))


def getCommand(drone, state, bounding_boxes, frame_center):
    if state == ControlStates.ON_GROUND:
        return drone.takeoff, (), state
    elif state == ControlStates.CENTER:
        biggest_box = sorted(bounding_boxes, key=lambda box: box.numpy()[0][2] * box.numpy()[0][3])
        print(biggest_box)
        command, arguments = proportional_centering(drone, biggest_box[0].numpy()[0][0], frame_center)
        return command, arguments, state
    else:
        drone.rc_control(0, 0, 0, 0)
        return drone.land, (), state


def sendCommand(command, arguments):
    return command(*arguments)


def sendCommandToDrone(drone, bounding_boxes, dimens, state):
    command, arguments, state = getCommand(drone, state, bounding_boxes, int(dimens[1] / 2))
    response = sendCommand(command, arguments) == 'ok'
    # response = str(state)
    print(bounding_boxes)
    """
    [tensor([[245., 200., 292., 306.],
        [ 70., 220., 115., 317.],
        [550., 150., 583., 227.],
        [360.,  83., 393., 165.],
        [236.,   0., 266.,  57.],
        [511., 201., 544., 274.],
        [518.,   0., 545.,  60.],
        [379., 332., 425., 440.],
        [451.,  76., 486., 155.],
        [ 38., 215.,  73., 297.],
        [435., 147., 461., 220.],
        [149., 415., 202., 476.]], device='cuda:0')]
    """

    print(dimens)  # DIMENS (480, 640, 3)

    # tello.Tello()

    return response, state


if __name__ == '__main__':
    drone, drone_state, start_time = init_drone()
    cap = cv2.VideoCapture('udp://' + drone.tello_ip + ':11111')
    # drone.streamon()
    # while True:
    #    command_successful, drone_state = sendCommandToDrone(drone, [(0, 2, 90, 5)], (480, 640, 3), drone_state,
    # start_time)
    # ret, frame = cap.read()
    # if ret:
    #    cv2.imshow("cap", frame)
    #    k = cv2.waitKey(1) & 0xFF
    #    if k == 27:
    #        break
    # cap.release()
    # cv2.destroyAllWindows()
