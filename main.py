from __future__ import print_function
from robolab_turtlebot import Turtlebot, get_time
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
    def __init__(self, turtle, exit:threading.Event):
        self.turtlebot = turtle
        self.lock = threading.Lock()
        self.stopped = False
        self.exit = exit
        self.turtlebot.register_bumper_event_cb(self.bumper_cb)

    def bumper_cb(self, msg):
        if msg.state == 1:
            self.stop()
            self.exit.set()
        
    def move(self,linear=0,angular=0):
        with self.lock:
            if self.stopped:
                return
            self.turtlebot.cmd_velocity(linear, angular)
    
    def stop(self):
        with self.lock:
            self.stopped = True
            self.turtlebot.cmd_velocity(0, 0)

class ImageProcessor:
    def __init__(self, turtle):
        self.turtlebot = turtle
    
    def get_image(self): # TODO: Type hint
        return self.turtlebot.get_rgb_image()

def exit_thread(exit:threading.Event, controler: TurtlebotController):
    start = get_time()

    while not exit.is_set():
        if get_time() - start > 20:
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

def image_thread(image_queue: Queue, gui_queue: Queue ,exit:threading.Event, processor: ImageProcessor):
    now = 0
    ticks = 0
    before = get_time()

    while not exit.is_set():
        now = get_time()
        image = processor.get_image()
        if image is not None:
            image_queue.put(image)
            gui_queue.put(image)
        if (now - before) > 1:
            print("Image ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1

def main_thread(image_queue: Queue, exit: threading.Event, controler: TurtlebotController):
    now = 0
    ticks = 0
    before = get_time()

    angular = 0.5
    t = get_time()

    while not exit.is_set():
        now = get_time()

        if now - t > 5:
            while get_time() - now < 1:
                controler.move(angular, angular)
            angular = -angular
            t = now
        
        controler.move(0, angular)
        
        if now - before > 1:
            print("Main ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1


def main():
    turtle = Turtlebot(rgb=True, pc=True)
    exit = threading.Event()
    controler = TurtlebotController(turtle, exit)
    processor = ImageProcessor(turtle)
    image_queue = Queue()
    gui_queue = Queue()

    threads = [
        threading.Thread(target=exit_thread, args=(exit,controler,)),
        threading.Thread(target=gui_thread, args=(gui_queue,exit,)),
        threading.Thread(target=image_thread, args=(image_queue, gui_queue,exit,processor,)),
        threading.Thread(target=main_thread, args=(image_queue,exit,controler,))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
    print("All threads have finished execution.")

if __name__ == "__main__":
    main()
