
from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate, get_time
import cv2
import threading
import math
import numpy as np
import ctypes
import time
 
# Initialization of threads for X11, important for GUI management in Linux
ctypes.CDLL("libX11.so.6").XInitThreads()
 
### POINT ###
"""
Point class
 
representing a single point in the image
"""
class Point:
    def __init__(self, x, y, dist):
        self.x = x
        self.y = y
        self.dist = dist
        self.dist_in_m = dist / 1000
 
    def __str__(self):
        return f"x:{self.x}, y:{self.y}, dist:{self.dist}"
 
 
CENTER_IMG = Point(320, 240, -1)  # Center point of the image frame
 
### IMAGE PROCESSING ###
"""
Image Processor class
 
handles image processing from the robot's RGB and Depth cameras.
"""
class ImageProcessor:
    # Convert RGB image from robot camera to HSV color space
    @staticmethod
    def get_image_hsv(turtle):
        img = turtle.get_rgb_image()
 
        if img is None:
            return None
 
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    # Create color masks for specified HSV ranges
    @staticmethod
    def get_hsv_channels(hsv):
        kernel = np.ones((3, 3), "uint8")
 
        masks = {
            "yellow": cv2.inRange(
                hsv,
                np.array([20, 100, 60], np.uint8),
                np.array([30, 255, 255], np.uint8),
            ),
            "red": cv2.inRange(
                hsv, np.array([0, 90, 60], np.uint8),
                np.array([8, 255, 255], np.uint8))
                + cv2.inRange(hsv,
                np.array([162, 90, 60], np.uint8),
                np.array([179, 255, 255], np.uint8),
            ),
            "blue": cv2.inRange(
                hsv,
                np.array([90, 100, 65], np.uint8),
                np.array([125, 255, 255], np.uint8),
            ),
            "green": cv2.inRange(
                hsv,
                np.array([40, 70, 60], np.uint8),
                np.array([75, 255, 255], np.uint8),
            ),
        }
 
        for key in masks:
            masks[key] = cv2.dilate(masks[key], kernel)
 
        return {
            key: cv2.bitwise_and(
                hsv,
                hsv,
                mask = masks[key]) for key in masks
            }
 
    # Detect and label objects (ball, gates) from provided HSV masks
    @staticmethod
    def labeling(img, res, robot):
        ball = None
        gates = []
 
        for key in res:
            # Convert to grayscale
            gray = cv2.cvtColor(res[key], cv2.COLOR_BGR2GRAY)
 
            # Binary mask for the given camera BGR image
            mask = cv2.threshold(
                gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
 
            # Find contours of the objects on image from the binary mask
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
 
            # Identifie the ball from the red mask and store it to the robot's 
            # memory
            if key == "red":
                ball, img = ImageProcessor.find_ball(
                    robot.turtle,contours,img,(191, 0, 255)
                )
                robot.ball = ball
 
            # if key == "green":
            #     img = ImageProcessor.draw_hull(
            #         contours,img,(0, 255, 0)
            #     )
 
            # if key == "yellow":
            #     ball, img = ImageProcessor.find_ball(
            #         turtle,contours,img,(100, 200, 200)
            #     )
 
            # Detect gates from blue mask
            if key == "blue":
                gates, img = ImageProcessor.find_gates(
                    robot, contours, img, (255, 0, 0)
                )
 
        return img, ball, gates
 
    # Identifie gates based on contours and mark them in the image
    @staticmethod
    def find_gates(robot, contours, img, color):
        gates = []
 
        for con in contours:
            x, y, width, height = cv2.boundingRect(con)
            area = cv2.contourArea(con)
 
            if 350 < area < 39000 and height / width > 2.3:
                tmp = Point(
                    int(x + width / 2),
                    int(y + height / 2),
                    ImageProcessor.get_distance(
                        robot.turtle, int(y + height / 2), int(x + width / 2)
                    ),
                )
 
                if tmp.dist_in_m < 0.2:
                    continue
 
                gates.append(tmp)
                img = cv2.rectangle(
                    img, (x, y), (x + width, y + height), color, 1)
 
        return gates, img
 
    # Detect ball based on contours and mark its position on the image
    @staticmethod
    def find_ball(turtle, contours, img, color):
        ball = None
 
        for con in contours:
            hull = cv2.convexHull(con)
            area = cv2.contourArea(con)
 
            if area < 450 or area > 55000:
                continue
 
            (x1, y1), (width, height), angle = cv2.minAreaRect(hull)
            (x, y), radius = cv2.minEnclosingCircle(hull)
 
            f1 = area / (width * height)
            f2 = area / (math.pi * radius**2)
 
            # Compute given red objects to detect only correct ball
            if f2 > 0.3 and ((f1 < f2 and 0.75 < f2 < 1.2)  
                    or (area > 4500)):   # area > 4000 or width > height * 1.3
                tmp = Point(int(x), int(y),
                    ImageProcessor.get_distance(turtle, y, x))
 
                if tmp.dist_in_m < 0.3: 
                    continue
 
                img = cv2.circle(img, (int(x), int(y)), int(radius), color, 1)
                ball = tmp
 
        return ball, img
 
    # Draw convex hull around provided contours for visualization
    @staticmethod
    def draw_hull(contours, img, color):
        for con in contours:
            hull = cv2.convexHull(con)
            area = cv2.contourArea(con)
 
            if 500 < area < 40000:
                img = cv2.drawContours(img, [hull], -1, color, 2)
 
        return img
 
    # Calculate distance from robot to point based on depth data
    @staticmethod
    def get_distance(turtle, y, x):
        x = int(x)
        y = int(y)
 
        pc = turtle.get_depth_image()
 
        if pc is None:
            return None
 
        # Crop and resize the depth image to correct it's scale with
        # image from the RGB camera
        pc = pc[95:400, 110:510]
        pc = cv2.resize(pc, (640, 480), interpolation = cv2.INTER_NEAREST)
        
        # Define the area where to look for object
        rect_x = 7
        rect_y = 3
 
        ans = -1
 
        # Choose the most precise distance to point
        for i in range(-rect_x, rect_x, 1):
            for j in range(-rect_y, rect_y, 1):
                if x + i < 0 or x + i > 639 or y + j < 0 or y + j > 479 \
                    or pc[y + j, x + i] < 0.1:
                    continue
 
                else:
                    if ans == -1 or pc[y + j, x + i] < ans:
                        ans = pc[y +j, x + i]
 
        return ans
         
 
### ROBOT  CONTROLLER ###
"""
Robot Controller class
 
main class to control robot movement and interaction
"""
class RobotController:
    def __init__(self):
        self.turtle = Turtlebot(pc = False, rgb = True, depth = True)
 
        self.turtle.register_bumper_event_cb(self.bumper_cb)
        self.turtle.register_button_event_cb(self.button_cb)
        self.bumper_triggered = False
        self.button_pressed = False
 
        self.result = None
        self.gates = []
        self.ball = None
        self.ball_found = False
 
        # Constant to hadle situations, when robot do not need to compute ball
        self.ball_ignore = False
 
        self.state = "ROTATE"
 
        self.count = 0
        self.phi_a = 0
        self.x_fw = 0
        self.side = 0
         
        # Define the minimum distance between poles for further calculations
        self.dist_between_poles = 0.7
 
        self.clear_goal = False
        self.complete_rotation = False
        self.d1 = 0
        self.d2 = 0
 
        self.count2 = 0
        self.exception = False
        self.gate_center = None
         
        self.mutex = threading.Lock()
 
    # Check if the robot should terminate operations
    def shut_down(self):
        return not self.turtle.is_shutting_down() and not self.bumper_triggered
 
    # Callback triggered by robot's bumper sensor event
    def bumper_cb(self, msg):
        if msg.state == 1:
            self.bumper_triggered = True
 
    # Callback triggered by pressing button on robot
    def button_cb(self, msg):
        if msg.state == 1:
            self.button_pressed = True
 
    # Send linear and angular velocity commands to robot
    def move(self, angular = 0, linear = 0):
        if not self.turtle.is_shutting_down():
            self.turtle.cmd_velocity(angular = angular, linear = linear)
 
    # Continuously process camera images for object detection and labeling,
    # handled with mutex for safe and correct threading
    def process_image(self):
        running = True
 
        while running:
            if self.bumper_triggered:
                break
 
            self.mutex.acquire()
 
            running = not self.turtle.is_shutting_down()
 
            if self.turtle.is_shutting_down():
                self.mutex.release()
                break
 
            self.handle_image()
 
            if self.result is None:
                self.mutex.release()
                continue
             
            # Create a circle im center of the whole image frame for better
            # vizualization
            self.result = cv2.circle(
                self.result, (CENTER_IMG.x, CENTER_IMG.y), 2, (255, 0, 120), 5
            )
 
            # Create a circle in the image of ball center for better vizualization
            if self.ball:
                self.result = cv2.circle(
                    self.result, (self.ball.x, self.ball.y), 2, (100, 0, 120), 5
                )
 
            # Create a circle in the image of gate center for better vizualization
            if len(self.gates) == 2:
                self.result = cv2.circle(
                    self.result,
                    (
                        (int((self.gates[1].x + self.gates[0].x) / 2)),
                        int((self.gates[0].y + self.gates[1].y) / 2),
                    ),
                    2,
                    (255, 0, 30),
                    5,
                )
 
            self.mutex.release()
             
            cv2.imshow("RESULT", self.result)
            cv2.waitKey(1)
 
    # Process camera images for object detection and labeling,
    # handled without mutex -> not safe, used only once
    def process_image_notsafe(self):
        running = True
 
        self.handle_image()
 
        if self.result is None:
            return
 
        # Create a circle im center of the whole image frame for better
        # vizualization
        self.result = cv2.circle(
            self.result, (CENTER_IMG.x, CENTER_IMG.y), 2, (255, 0, 120), 5
        )
 
        # Create a circle in the image of ball center for better vizualization
        if self.ball:
            self.result = cv2.circle(
                self.result, (self.ball.x, self.ball.y), 2, (100, 0, 120), 5
            )
 
        # Create a circle in the image of gate center for better vizualization
        if len(self.gates) == 2:
            self.result = cv2.circle(
                self.result,
                (
                    (int((self.gates[1].x + self.gates[0].x) / 2)),
                    int((self.gates[0].y + self.gates[1].y) / 2),
                ),
                2,
                (255, 0, 30),
                5,
            )
 
    # Handle a single image frame for object detection
    def handle_image(self):
        hsv = ImageProcessor.get_image_hsv(self.turtle)
 
        if hsv is None:
            return
 
        res = ImageProcessor.get_hsv_channels(hsv)
 
        # + res["yellow"] + res["green"]
        self.result = res["blue"] + res["red"]
        self.result, self.ball, self.gates = ImageProcessor.labeling(
            self.result, res, self
        )
 
    # Move robot forward for specific distance with directional adjustment
    def go_forward(self, dist, side):
        self.go_forward_unlimited(min(dist, 1.5), side)
 
    # Move robot forward indefinitely with ball avoidance
    def go_forward_unlimited(self, dist, side):
        self.move()
        PARAM = dist * 2.2
        t = get_time()
 
        while (get_time() - t < max(PARAM, 0.4)) and self.shut_down():
            # Unsafely process the image, while moving to next robot position,
            # for detection of the ball on robot's route and than avoid it
            self.process_image_notsafe()
 
            if self.ball and self.ball.dist_in_m < 0.8:
                t2 = get_time()
 
                while (get_time() - t2 < 2.15) and self.shut_down():
                    if side == 1:
                        self.move(angular = 0.77, linear = 0.4)
                    else:
                        self.move(angular = -0.77, linear = 0.4)
 
                    self.ball_ignore = True
 
                self.move(angular = 0, linear = 0)
 
            else:
                if dist <= 0.2:
                    self.move(linear = 0.3)
                else:
                    self.move(linear = 0.6)
 
        self.move()    
 
    # Rotate robot by specified degrees in the given direction
    def turn_deg(self, phi, side):
        self.move()
        PARAM = phi / 90 * 4.25
        t = get_time()
 
        while (get_time() - t < PARAM) and self.shut_down():
            self.move(angular = side * 0.7)
 
        self.move()
 
    # Continuously manage robot's movement based on its state
    def process_movement(self):
        running = True
 
        while running:
            if self.bumper_triggered:
                break
 
            self.mutex.acquire()
 
            running = not self.turtle.is_shutting_down()
 
            if self.turtle.is_shutting_down():
                self.mutex.release()
                break
 
            self.handle_state()
            self.mutex.release()
            time.sleep(0.05)
 
    # Manage robot behavior according to its current operational state
    def handle_state(self):
 
        if (self.state == "LOOK" or self.state == "FINDGATE") \
            and len(self.gates) == 2:
            if self.complete_rotation:
                self.state = "ROTATE"
                return
 
            elif not self.exception:
                self.state = "FINDGATE"
 
        print(f"-- {self.state} --")
         
        ############################ ROBOT MOVEMENT ############################
 
        # LOOK: Robot rotates to search for gates and the ball
        if self.state == "LOOK":
            self.move(angular = 0.5)
 
        # EXCEPTION: Handles scenarios when gates distances vary significantly
        elif self.state == "EXCEPTION":
            if len(self.gates) != 2:
                self.move(angular = 0.4)
 
            else:
                self.move()
 
                side_1 = a = 0
                # Multiplication by 0.98 is made for more precise calculation of
                # depth to the gates, due to depth camera's angle
                a = max(self.gates[0].dist, self.gates[1].dist) * 0.98
 
                # Decide on which side of the gate is current robot position
                if self.gates[1].dist > self.gates[0].dist:
                    side_1 = 1
                else:
                    side_1 = -1
 
                self.turn_deg(50, side_1)
                self.go_forward_unlimited(1 * 0.85 * a, side_1)
                 
                self.state = "GOAL_EXC"
                self.move()
 
        # GOAL_EXC: Moves robot towards the ball after handling an 
        # exception state
        elif self.state == "GOAL_EXC":
            if self.ball:
                # Center the robot on ball center
                if CENTER_IMG.x - self.ball.x < -7:
                    self.move(angular = -0.2 -
                              abs(CENTER_IMG.x - self.ball.x) * 0.003)
 
                elif CENTER_IMG.x - self.ball.x > 7:
                    self.move(angular = 0.2 +
                              abs(CENTER_IMG.x - self.ball.x) * 0.003)
 
                else:
                    # Go forward to score goal
                    t = get_time()
 
                    while (t - get_time()) < 3 and self.shut_down():
                        self.move(linear = 0.9)
 
                    self.state = "DEATH"
 
            else:
                self.move(angular = 0.5)
 
        # FINDGATE: Adjusts robot orientation to center itself between
        # detected gates
        elif self.state == "FINDGATE":
            angular = 0
 
            if len(self.gates) != 2:
                self.state = "LOOK"
                return
 
            if self.count2 >= 10:
                self.count2 = 0
                self.gate_center = None
 
            if self.gate_center is None:
                self.gate_center = (self.gates[0].x + self.gates[1].x) / 2
 
            # Check if current robot position will be fine for further movement
            if abs(self.gates[0].dist_in_m - self.gates[1].dist_in_m) > 2000:
                self.exception = True
                self.state = "EXCEPTION"
                return

            if self.ball and len(self.gates) == 2:
                self.state = "ZALUPA"
                return
 
            # Center the robot on gate center
            if CENTER_IMG.x - self.gate_center < -5:
                angular = -0.2 - abs(CENTER_IMG.x - self.gate_center) * 0.003
 
            elif CENTER_IMG.x - self.gate_center > 5:
                angular = 0.2 + abs(CENTER_IMG.x - self.gate_center) * 0.003
 
            else:
                # Move the robot forward if the distance to gate is to big for better 
                # further calculations
                if self.gates[0].dist_in_m > 2.5 and self.gates[1].dist_in_m > 2.5:
                    t4 = get_time()
 
                    while (get_time() - t4 < 0.5) and self.shut_down():
                        self.move(linear = 0.4)
 
                self.state = "CALCULATE"
                self.count2 = 0
                self.gate_center = None
 
            self.move(angular = angular)
            self.count2 += 1
 
        # CALCULATE: Computes optimal movement path based on distances
        # to detected gates
        elif self.state == "CALCULATE":
            self.move()
 
            if self.count == 0:
                self.phi_a = 0
                self.x_fw = 0
                self.side = 0
 
            if len(self.gates) != 2:
                self.state = "FINDGATE"
                self.count = 0
                return
 
            self.d1, self.d2 = self.gates[1].dist_in_m, self.gates[0].dist_in_m
            print(self.d1, self.d2, "============")
 
            # Decide on which side of the gate is current robot position.
            # Multiplication by 0.98 is made for more precise calculation of
            # depth to the gates, due to depth camera angle
            if self.gates[0].x > self.gates[1].x:
                if self.d2 > self.d1:
                    #self.d2 = self.d2 * 0.98
                    self.side = -1
                else:
                    self.d1, self.d2 = self.d1, self.d2
                    self.side = 1
            else:
                if self.d2 > self.d1:
                    #self.d2 = self.d2 * 0.98
                    self.side = 1
                else:
                    self.d1, self.d2 = self.d1, self.d2
                    self.side = -1

            print(self.d1, self.d2, self.side)
 
            # Calculate numerator and denominator as the part of the arctan 
            # expression to determine the angle phi using the law of cosines
            numerator = abs(self.d1**2 - self.d2**2) / (2 * self.dist_between_poles)
            denominator = np.sqrt(self.d1**2 -
                (((self.d1**2 - self.d2**2 +
                 (self.dist_between_poles**2)) ** 2) /
                  (4 * (self.dist_between_poles**2))))
 
            # Check if calculated values are invalid (Not a Number or division
            # by zero), adjust pole distance slightly to avoid math errors and
            # reattempt gate detection
            if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
                self.dist_between_poles += 0.05
                self.state = "FINDGATE"
                self.count = 0
                return
 
            # Calculate angle between gate center and robot
            phi = np.arctan(numerator / denominator)
 
            # Estimate the forward distance for robot to move based on the 
            # average distance to poles and angle phi
            z = ((self.d1 + self.d2) / 2) * np.tan(phi)
 
            # Check, that forward distance is not to large
            self.x_fw += min(z, 3,5)
 
            self.phi_a += phi
            self.count += 1
 
            # Once enough valid values have been collected, average results
            # for best accuracy
            if self.count >= 25:
                self.phi_a /= self.count
                self.x_fw /= self.count
                self.count = 0
 
                self.move()
 
                # Validate calculated output and decide if robot can make a goal
                # from it's current positon, otherwise proceed to movement
                if self.x_fw <= 0.08 and abs(self.d1 - self.d2) < 0.01 \
                    and self.ball is None:
                    self.clear_goal = True
                    self.state = "GOAL"
 
                else:
                    self.state = "MOVE"
 
        # MOVE: Executes calculated forward movement value and rotation towards
        # target location
        elif self.state == "MOVE":
            # Rotate the robot and move it to new better position
            self.turn_deg(90, self.side)

            if self.side == 0:
                pass
            else:
                self.go_forward(self.x_fw, self.side)
             
            self.x_fw = 0
            self.phi_a = 0
            self.ball_found = False
            self.dist_between_poles = 0.7
             
            self.state = "ROTATE"
 
        # ROTATE: Rotates robot to locate ball and gates,
        # determining next step based on their positions
        elif self.state == "ROTATE":
            ball_for_zalupa = False
            self.move()
            self.complete_rotation = True
            dist_gt = 0
            angular = 0
 
            if len(self.gates) == 2:
                dist_gt = (
                    max(self.gates[1].dist, self.gates[0].dist) * 0.98
                    + min(self.gates[1].dist, self.gates[0].dist)) / 2000
 
            if self.ball is None and len(self.gates) == 2 \
                and max(self.gates[0].x, self.gates[1].x) < 540:
                self.turn_deg(12, 1)
                if self.ball:
                    ball_for_zalupa = True             

            if self.ball is None and ball_for_zalupa == False:
                if len(self.gates) == 2:
                    self.complete_rotation = False
 
                self.state = "LOOK"
                return
 
            #if (len(self.gates) == 2 and (
            #    ((self.gates[0].x + self.gates[1].x) / 2) - self.ball.x) > 100):
 
            #    self.state = "FINDGATE"
            #    self.complete_rotation = False
            #    self.ball_ignore = False
            #    return
            
 
            # Calculate the parameter of how precise should the ball be to the 
            # gate center to make a goal based on distance from robot
            if len(self.gates) == 2:
                PARAM = max(int((4.8 * (dist_gt ** 2))), 5)
 
             # Center the robot on ball center
            if CENTER_IMG.x - self.ball.x < -7:
                angular = -0.2 - abs(CENTER_IMG.x - self.ball.x) * 0.003
 
            elif CENTER_IMG.x - self.ball.x > 7:
                angular = 0.2 + abs(CENTER_IMG.x - self.ball.x) * 0.003
 
            else:
                if (
                    len(self.gates) == 2
                    and (
                        (CENTER_IMG.x - self.ball.x) > -PARAM
                        and (CENTER_IMG.x - ((self.gates[0].x + self.gates[1].x) / 2))
                        > -PARAM
                    )
                    and (
                        (CENTER_IMG.x - self.ball.x) < PARAM
                        and (CENTER_IMG.x - ((self.gates[0].x + self.gates[1].x) / 2))
                        < PARAM
                    )
                ):
                    self.state = "GOAL"
 
                else:
                    self.complete_rotation = False
                    #self.state = "FINDGATE"
                    self.state = "ZALUPA"

                    #if self.gates[0].x < self.ball.x < self.gates[1].x \
                    #    or self.gates[1].x < self.ball.x < self.gates[0].x:
                    #    self.state = "ZALUPA"
 
            self.move(angular = angular)
            return

        elif self.state == "ZALUPA":
            if len(self.gates) < 2 and self.ball:
                self.state = "FINDGATE"
                return

            param = 0.2

            if self.ball.x < int((self.gates[0].x + self.gates[1].x)/2):
                self.side = 1
                self.turn_deg(90, self.side)
                self.go_forward(param, self.side)

            else:
                self.side = -1
                self.turn_deg(90, self.side)
                self.go_forward(param, self.side)

            self.state  = "ROTATE"

 
        # GOAL: Robot moves forward directly towards the ball and attempts
        # goal action
        elif self.state == "GOAL":
            PARAM = 0
 
            if self.ball is None and self.clear_goal == True:
                t5 = get_time()
 
                while get_time() - t5 < 1 and self.shut_down():
                    self.move(linear = -0.35)
 
                self.state = "ROTATE"
                self.clear_goal = False
                return
 
            if self.ball is not None and len(self.gates) == 2:
                # Calculate the parameter of how far should the robot go
                # to make a goal
                PARAM = self.ball.dist_in_m * 1.2
                t3 = get_time()
 
                while (get_time() - t3 < PARAM) and self.shut_down():
                    self.move(linear = 0.93, angular = (
                        CENTER_IMG.x - self.ball.x) * 0.003)
 
                self.state = "DEATH"
 
            else:
                self.state = "ROTATE"
 
    # Monitor exit conditions
    def exit_thread(self):
        running = True
 
        while running:
            self.mutex.acquire()
            running = not self.turtle.is_shutting_down()
 
            if self.turtle.is_shutting_down():
                self.mutex.release()
                break
 
            if self.bumper_triggered:
                print("\nEND: Bumper exit")
                self.turtle.play_sound(sound_id = 4)
                break
 
            if self.turtle.is_shutting_down():
                print("\nEND: Ctrl + C exit")
                self.turtle.play_sound(sound_id = 6)
                break
 
            self.mutex.release()
 
        self.mutex.release()
 
    # Initialize and run image processing, movement,
    # and exit monitoring threads
    def run(self):
        print("Starting ...")
 
        self.turtle.play_sound(sound_id = 1)
 
        # Wait till the button on the robot is pressed to start the program
        #while not self.button_pressed:
        #    pass
         
        image_thread = threading.Thread(target=self.process_image)
        movement_thread = threading.Thread(target=self.process_movement)
        exit_thread = threading.Thread(target=self.exit_thread)
 
        image_thread.start()
        movement_thread.start()
        exit_thread.start()
 
        image_thread.join()
        print("IMAGE THREAD closed")
        movement_thread.join()
        print("MOVE THREAD closed")
        exit_thread.join()
        print("DEATH THREAD closed")
 
        print("Ending ...")
         
        self.turtle.play_sound(sound_id = 2)
 
 
def main():
    robot = RobotController()
    robot.run()
 
 
if __name__ == "__main__":
    main()
     
    
