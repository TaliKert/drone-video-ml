import time

from easytello import tello

from enum import Enum


class ControlStates(Enum):
    ON_GROUND = 1
    CENTER = 2
    LAND = 3


def init_drone():
    drone = tello.Tello()
    drone.takeoff()
    send_rc(drone, 0, 0, 0, 0)
    drone.up(150)
    drone.send_command('streamon')
    return drone, ControlStates.CENTER, time.time()


def send_rc(drone, a, b, c, d):
    command = 'rc {} {} {} {}'.format(a, b, c, d)
    print('Sending command: {}'.format(command))
    drone.socket.sendto(command.encode('utf-8'), drone.tello_address)


# P - controller
def proportional_checking(drone, bounding_box_center, frame_center, size, wanted_size):
    KPy = 0.1
    KPf = 0.00015
    error_yaw = frame_center - bounding_box_center
    error_forward = wanted_size - size
    """
    Send RC control via four channels.
        a: left/right (-100~100)
        b: forward/backward (-100~100)
        c: up/down (-100~100)
        d: yaw (-100~100)
    """
    return send_rc, (drone, 0, int(KPf*error_forward), 0, -int(KPy * error_yaw))


def getCommand(drone, state, bounding_boxes, frame_center):
    if state == ControlStates.ON_GROUND:
        return drone.takeoff, (), state
    elif state == ControlStates.CENTER:
        biggest_box = sorted(bounding_boxes, key=lambda box: box.numpy()[0][2] * box.numpy()[0][3])
        size = biggest_box[0].numpy()[0][2] * biggest_box[0].numpy()[0][3]
        command, arguments = proportional_checking(drone, biggest_box[0].numpy()[0][0], frame_center, size, 400000.0)
        return command, arguments, state
    else:
        drone.rc_control(0, 0, 0, 0)
        return drone.land, (), state


def sendCommand(command, arguments):
    return command(*arguments)


def sendCommandToDrone(drone, bounding_boxes, dimens, state):
    command, arguments, state = getCommand(drone, state, bounding_boxes, int(dimens[1] / 2))
    response = sendCommand(command, arguments) == 'ok'
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

    return response, state
