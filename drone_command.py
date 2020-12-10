from easytello import tello

def getCommand(bounding_boxes):
    return None


def sendCommand(command):
    return None


def sendCommandToDrone(bounding_boxes, dimens):



    command = getCommand(bounding_boxes)
    sendCommand(command)
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

    print(dimens) # DIMENS (480, 640, 3)

    # tello.Tello()

    return None