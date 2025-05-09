from __future__ import print_function
from robolab_turtlebot import Turtlebot, get_time # type: ignore
import cv2
import threading
import numpy as np
import ctypes
from queue import Queue

ctypes.CDLL('libX11.so.6').XInitThreads()

### CLASS DEFINITION ###

class SharedData:
    def __inti__(self, rgb_image=None, depth_image=None):
        self.rgb_image = rgb_image
        self.depth_image = depth_image

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
        
    def move(self,linear=0,angular=0):
        with self.lock:
            if self.stopped or self.turtlebot.is_shutting_down():
                return
            self.turtlebot.cmd_velocity(linear, angular)
    
    def rotate(self, angle, speed):
        self.turtlebot.reset_odometry()
        t = get_time()
        while get_time() - t < 10:
            print("Rotating: ", self.turtlebot.get_odometry()[2], "angle: ", angle*np.pi/180)
            self.move(0, np.sign(angle)*speed)

    def stop(self):
        self.stopped = True
        with self.lock:
            self.move(0, 0)
        self.exit.set()
        # with self.lock:
        #     self.stopped = True
        #     self.exit.set()
        #     if not self.turtlebot.is_shutting_down():
        #         self.turtlebot.cmd_velocity(0, 0)

class ImageProcessor:
    def __init__(self):
        self.turtlebot = Turtlebot(rgb=True, pc=True)
        self.turtlebot.wait_for_rgb_image()
        self.turtlebot.wait_for_depth_image()
    
    def get_image(self):
        return self.turtlebot.get_rgb_image()
    
    def get_depth(self):
        return self.turtlebot.get_depth_image()

### THREAD FUNCTIONS ###

def image_thread(image_queue:Queue, gui_queue:Queue, exit:threading.Event, processor:ImageProcessor):
    now = 0
    ticks = 0
    before = get_time()

    while not exit.is_set():
        now = get_time()
        image = processor.get_image()
        depth = processor.get_depth()
        
        if image is not None and depth is not None:
            image_queue.put(image)
            gui_queue.put(image)
        
        if now - before > 1:
            print("Image ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1

def main_thread(image_queue:Queue, controler:TurtlebotController):
    now = 0
    ticks = 0
    before = get_time()



    while not controler.exit.is_set():
        now = get_time()
        
        controler.rotate(45, 0.5)

        

        if now - before > 1:
            print("Main ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1

def exit_thread(controler:TurtlebotController):
    start = get_time()

    while not controler.exit.is_set():
        if get_time() - start > 20 or controler.turtlebot.is_shutting_down():
            controler.stop()

def gui_thread(gui_queue:Queue, exit:threading.Event):
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
            # print("GUI ticks: ", ticks)
            ticks = 0
            before = now
        ticks += 1

def main():
    turtle = Turtlebot()

    exit = threading.Event()
    controler = TurtlebotController(turtle, exit)
    processor = ImageProcessor()
    image_queue = Queue()
    gui_queue = Queue()

    threads = [
        threading.Thread(target=exit_thread, args=(controler,)),
        threading.Thread(target=gui_thread, args=(gui_queue, exit,)),
        threading.Thread(target=image_thread, args=(image_queue, gui_queue, exit, processor)),
        threading.Thread(target=main_thread, args=(image_queue,controler,))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    
    print("All threads have finished execution.")

if __name__ == "__main__":
    main()
