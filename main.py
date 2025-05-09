from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate, get_time
import cv2
import threading
import math
import numpy as np
import ctypes
from queue import Queue

ctypes.CDLL('libX11.so.6').XInitThreads()

class SharedData:
    def __inti__(self):
        pass

class TurtlebotController:
    def __init__(self):
        self.turtlebot = Turtlebot(pc=False, rgb=False)
        self.lock = threading.Lock()
        self.stopped = False
    
    def move(self,linear=0,angular=0):
        with self.lock:
            if self.stopped:
                return
            self.turtlebot.move(linear, angular)
    
    def stop(self):
        with self.lock:
            self.stopped = True
            self.turtlebot.move(0, 0)

class ImageProcessor:
    def __init__(self):
        self.turtlebot = self.turtlebot = Turtlebot(pc=True, rgb=True)
    
    def get_image(self): # TODO: Type hint
        return self.turtlebot.get_image()

def exit_thread(exit:threading.Event):
    start = get_time()
    controler  = TurtlebotController()
    while not exit.is_set():
        if get_time() - start > 10:
            exit.set()
            controler.move(0, 0)


def gui_thread(gui_queue: Queue, exit:threading.Event):
    now = 0
    ticks = 0
    before = get_time()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    while not exit.is_set():
        now = get_time()
        if not gui_queue.empty():
            data = gui_queue.get()
            cv2.imshow("Image", data)
            cv2.waitKey(1)
        if now - before > 1:
            print("GUI ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1

def image_thread(image_queue: Queue, exit:threading.Event):
    ImageProcessor = ImageProcessor()

    now = 0
    ticks = 0
    before = get_time()

    while not exit.is_set():
        now = get_time()
        image = ImageProcessor.get_image()
        if image is not None:
            image_queue.put(image)
        if (now - before) > 1:
            print("Image ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1

def main_thread(image_queue: Queue, gui_queue: Queue, exit: threading.Event):
    now = 0
    ticks = 0
    before = get_time()

    controler = TurtlebotController()
    angular = 0.2
    t = get_time()

    while not exit.is_set():
        now = get_time()
        
        if not image_queue.empty():
            data = image_queue.get()
            gui_queue.put(data)
        
        if t - get_time() > 5:
            angular = -angular
            t = get_time()
        
        controler.move(0, angular)
        
        if now - before > 1:
            print("Main ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1


def main():
    image_queue = Queue()
    gui_queue = Queue()
    exit = threading.Event()

    threads = [
        threading.Thread(target=exit_thread, args=(exit,)),
        threading.Thread(target=gui_thread, args=(gui_queue,exit,)),
        threading.Thread(target=image_thread, args=(image_queue,exit,)),
        threading.Thread(target=main_thread, args=(image_queue, gui_queue,exit,))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
    print("All threads have finished execution.")

if __name__ == "__main__":
    main()
