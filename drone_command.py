import time

from easytello import tello

from enum import Enum

ERROR_BOUNDARY = 50


class ControlStates(Enum):
    ON_GROUND = 1
    CENTER = 2
    LAND = 3


def init_drone():
    return tello.Tello(), ControlStates.ON_GROUND, time.time()


def proportional_centering(drone, bounding_box_center, frame_center):
    KP = 0.5
    error = frame_center - bounding_box_center
    """
    Send RC control via four channels.
        a: left/right (-100~100)
        b: forward/backward (-100~100)
        c: up/down (-100~100)
        d: yaw (-100~100)
    """
    return drone.rc_control, (KP * error, 0, 0, 0)


def getCommand(drone, state, bounding_boxes, frame_center):
    if state == ControlStates.ON_GROUND:
        return drone.takeoff, (), state
    elif state == ControlStates.CENTER:
        biggest_box = sorted(bounding_boxes, key=lambda x, y, w, h: w * h)
        return proportional_centering(drone, biggest_box[0], frame_center), state
    else:
        return drone.land, (), state


def sendCommand(command, arguments):
    return command(*arguments)


def sendCommandToDrone(drone, bounding_boxes, dimens, state, start_time):
    if time.time() - start_time > 5:
        state = ControlStates.LAND
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

    # tello.Tello()

    return response, state

if __name__ == '__main__':
    drone, drone_state, start_time = init_drone()
    command_successful, drone_state = sendCommandToDrone(drone, [(0, 2, 90, 5)], (480, 640, 3), drone_state, start_time)
