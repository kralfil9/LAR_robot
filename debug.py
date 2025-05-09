from __future__ import print_function
import cv2
import numpy as np
from robolab_turtlebot import Turtlebot

def main():
    turtle = Turtlebot(rgb=True,pc=True)

    print(type(turtle.get_rgb_image()))

if __name__ == '__main__':
    main()
