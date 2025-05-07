from __future__ import print_function
from robolab_turtlebot import Turtlebot, Rate, get_time
import cv2
import threading
import math
import numpy as np
import ctypes

ctypes.CDLL('libX11.so.6').XInitThreads()

BUMPER = False
CENTER_IMG = [240, 320]
ball_found = False
complete_rotation = False
ignore_ball = False


def bumper_callback(msg):
    global BUMPER
    if msg.state == 1:
        BUMPER = True

def process_labels(img, res_blue, res_green, res_red, res_yellow, turtle):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    res_masks = {
        "red": res_red,
        "blue": res_blue,
        "green": res_green,
        "yellow": res_yellow,
    }

    contours_by_color = {}
    for color, res in res_masks.items():
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        binary_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours_by_color[color] = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    ball = None
    gates = []
    global ball_found
  
    # Process each color
    for color, contours in contours_by_color.items():
        for contour in contours:
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(contour)
            if area > 250 and area < 50000:    
                if color == "red" and area > 1500 and area < 50000:
                    img = cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
                    (x, y), (width, height), angle = cv2.minAreaRect(hull)
                    (x, y), radius = cv2.minEnclosingCircle(hull)

                    f1 = area / (width * height)
                    f2 = area / (3.14 * radius ** 2)
                    if (f2 > 0.3 and ((f1 < f2 and 0.5 < f2 < 1.2) or (area > 3500))):
                        ball = (x, y, get_distance(turtle, y, x))
                        img = cv2.drawContours(img, [hull], -1, (191, 0, 255), 1)
                elif color == "blue":
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    if 265 < area < 19000 and w < 80 and (h/w > 2.5):                          
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                        temp = (int(y + h / 2), int(x + w / 2), get_distance(turtle, int(y + h / 2), int(x + w / 2)))
                        if ((ball is not None) and temp[2] >= 1500):
                            ball_found = True
                            gates.append((int(y + h / 2), int(x + w / 2), get_distance(turtle, int(y + h / 2), int(x + w / 2))))
                        elif ((temp[2] <= 1500) and ((ball is None) or ignore_ball == True)) :
                            gates.append((int(y + h / 2), int(x + w / 2), get_distance(turtle, int(y + h / 2), int(x + w / 2))))
                        elif (temp[2] >= 1500 and (ball_found == True or ignore_ball == True)):   # maybe add timer to reset ball_found -> False
                            gates.append((int(y + h / 2), int(x + w / 2), get_distance(turtle, int(y + h / 2), int(x + w / 2))))
                        
                elif color == "green":
                    if 500 < area < 18000:
                        color_map = {"green": (0, 255, 0)}
                        mg = cv2.drawContours(img, [hull], -1, color_map.get(color, (0, 255, 0)), 2)

                elif color == "yellow":
                    if 500 < area < 15000:
                        color_map = {"yellow": (100, 200, 200)}
                        mg = cv2.drawContours(img, [hull], -1, color_map.get(color, (0, 255, 0)), 2)

    return img, ball, gates

def get_hsv_image(turtle):
    img = turtle.get_rgb_image()
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def get_hsv_masks(hsv):
    kernel = np.ones((3, 3), "uint8")

    yellow_mask = cv2.inRange(hsv, np.array([21, 100, 255], np.uint8), np.array([29, 255, 255], np.uint8)) #Disabled
    red_mask = cv2.inRange(hsv, np.array([0, 90, 60], np.uint8), np.array([8, 255, 255], np.uint8))
    red_mask2 = cv2.inRange(hsv, np.array([162, 90, 60], np.uint8), np.array([179, 255, 255], np.uint8))
    blue_mask = cv2.inRange(hsv, np.array([80, 90, 75], np.uint8), np.array([125, 255, 255], np.uint8))    ###BRUH, after changes might be fine
    green_mask = cv2.inRange(hsv, np.array([40, 70, 255], np.uint8), np.array([75, 255, 255], np.uint8)) #Disabled, 60

    red_mask = cv2.dilate(red_mask, kernel)
    red_mask2 = cv2.dilate(red_mask2, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)
    yellow_mask = cv2.dilate(yellow_mask, np.ones((3, 3), "uint8"))
    green_mask = cv2.dilate(green_mask, kernel)

    res_masks = {
        "red": cv2.bitwise_and(hsv, hsv, mask=(red_mask + red_mask2)),
        "blue": cv2.bitwise_and(hsv, hsv, mask=blue_mask),
        "yellow": cv2.bitwise_and(hsv, hsv, mask=yellow_mask),
        "green": cv2.bitwise_and(hsv, hsv, mask=green_mask)
    }

    return res_masks

def get_distance(turtle, y, x):
    global CENTER_IMG

    pc = turtle.get_depth_image()
    if pc is None:
        return None
    pc = pc[70:400, 185:635]
    pc = cv2.resize(pc, (640, 480), interpolation=cv2.INTER_NEAREST)

    centy, centx = pc.shape
    CENTER_PC = [int(centy / 2), int(centx / 2)]

    b = np.zeros([pc.shape[0], pc.shape[1], 3])
    b[:, :, 0] = pc[:, :]
    b[:, :, 1] = pc[:, :]
    b[:, :, 2] = pc[:, :]

    diff = [CENTER_IMG[0] - y, CENTER_IMG[1] - x]

    min_distance = 3200
    for i in range(-3, 3):
        for j in range(-3, 3):
            if not (0 < int(x + i) < 2 * CENTER_PC[1]) or not (0 < int(y + j) < 2 * CENTER_PC[0]):
                continue
            data = pc[int(y + j), int(x + i)]
            if 5 < data < 3200 and data < min_distance:
                min_distance = data

    return np.percentile(min_distance, 10)

def turn_deg(turtle, phi, side):
    turtle.cmd_velocity(angular=0, linear = 0)
    PARAM = phi/90 * 4.25
    t = get_time()
    while (get_time() - t < PARAM):
        turtle.cmd_velocity(angular = side * 0.7)
    turtle.cmd_velocity(angular = 0, linear = 0)

def go_fw(turtle, dist, side):
    turtle.cmd_velocity(angular = 0, linear = 0)
    global ignore_ball

    dist = min (dist, 1.5)
    PARAM = dist*2
    direction = dist/abs(dist)

    t = get_time()
    while (get_time() - t < max(PARAM, 0.5)):
        hsv = get_hsv_image(turtle)
        if hsv is None:
            continue

        res_masks = get_hsv_masks(hsv)
        result = sum(res_masks.values())
        result, ball, gates = process_labels(result, res_masks["blue"], res_masks["green"], res_masks["red"], res_masks["yellow"], turtle)
        cv2.imshow("result", result)

        if ball and ball[2]/1000 < 0.8:
            t2 = get_time()
            while(get_time() - t2 < 3):
                cv2.imshow("result", result)
                if side == 1:
                    turtle.cmd_velocity(angular = 0.65, linear = 0.4)
                else:
                    turtle.cmd_velocity(angular = -0.65, linear = 0.4)
                ignore_ball = True
            turtle.cmd_velocity(angular = 0, linear = 0)
        else:
            if dist <= 0.2:
                turtle.cmd_velocity(linear = 0.3)
            else:
                turtle.cmd_velocity(linear = 0.6)
    turtle.cmd_velocity(angular = 0, linear = 0)

    
def main_thread(turtle):
    global CENTER_IMG
    global ball_found
    global complete_rotation
    global ignore_ball

    rate = Rate(10000)
    side = 0
    count = 0
    phi_a = 0
    x_fw = 0
    count2 =0
    dist_between_poles = 0.7
    gate_center = None
    clear_goal = False

    state = "FINDGATE"

    while not turtle.is_shutting_down():
        print (state)

        if BUMPER:
            break

        hsv = get_hsv_image(turtle)
        if hsv is None:
            continue

        res_masks = get_hsv_masks(hsv)
        result = sum(res_masks.values())

        #for i in range(1, 5, 2):
        #    result = cv2.bilateralFilter(result, 7, 150, 150)
        #    result = cv2.GaussianBlur(result, (i, i), 0)

        result, ball, gates = process_labels(result, res_masks["blue"], res_masks["green"], res_masks["red"], res_masks["yellow"], turtle)
        centy, centx, _ = result.shape
        CENTER_IMG = [int(centy / 2), int(centx / 2)]

        result = cv2.circle(result, (CENTER_IMG[1], CENTER_IMG[0]), 2, (42, 42, 100), 5)

        if ball is not None:
            result = cv2.circle(result, (int(ball[0]), int(ball[1])), 2, (0, 30, 255), 5)

        if state != "ROTATE" and state != "MOVE" and state != "DEATH" and state != "GOAL" and len(gates) == 2:
            if complete_rotation == True:
                state = "ROTATE"
                continue
            else:
                state = "FINDGATE"
            result = cv2.circle(result, ((int((gates[1][1] + gates[0][1]) / 2)), 240), 2, (255, 0, 30), 5)

        cv2.imshow("result", result)

        # State-based action
        if state == "LOOK" and not turtle.is_shutting_down():
            cv2.imshow("result", result)
            turtle.cmd_velocity(angular = 0.6)


        if state == "FINDGATE" and not turtle.is_shutting_down():
            turtle.cmd_velocity(angular = 0, linear = 0)
            cv2.imshow("result", result)

            if len(gates) < 2:
                state = "LOOK"
                continue

            # remember the value of gate center for 10 interations
            if count2 >= 10:
                count2 = 0
                gate_center = None

            if len(gates) == 2:
                if ((gate_center is None) or count2 == 0):
                    gate_center = (gates[1][1] + gates[0][1]) / 2
            
                if CENTER_IMG[1] - gate_center < -5:
                    turtle.cmd_velocity(angular = -0.2 - abs(CENTER_IMG[1] - gate_center) * 0.003)
                elif CENTER_IMG[1] - gate_center > 5:
                    turtle.cmd_velocity(angular = 0.2 + abs(CENTER_IMG[1] - gate_center) * 0.003)
                else:
                    # ride closer to gate
                    if (gates[0][2]/1000.0 > 2.5 and gates[1][2]/1000.0 > 2.5):
                        t4 = get_time()
                        while (get_time() - t4 < 0.5):
                            turtle.cmd_velocity(linear = 0.4)
                    state = "CALCULATE"
                    turtle.cmd_velocity(angular = 0, linear = 0)
            count2 += 1


        if state == "CALCULATE" and not turtle.is_shutting_down():

            if count == 0:
                phi_a = 0
                x_fw = 0
                side = 0

            if gates is None or len(gates) < 2:
                state = "FINDGATE"
                count = 0 
                continue
    
            d1, d2 = gates[1][2] / 1000, gates[0][2] / 1000
            if gates[1][2] > gates[0][2]:
                d1, d2 = d2, d1
                side = 1
            else: 
                side = -1
                
            numerator = abs(d1**2 - d2**2) / (2 * dist_between_poles)
            denominator = np.sqrt(d1**2 - (((d1**2 - d2**2 + (dist_between_poles**2))**2) / (4 * (dist_between_poles**2))))

            # check if values are correct
            if (np.isnan(numerator) or np.isnan(denominator) or denominator == 0):
                dist_between_poles += 0.05
                state = "FINDGATE"
                print(dist_between_poles, "--------------------")
                print("NUM: ",numerator," DEN: ",denominator)
                count = 0
                continue

            phi = np.arctan(numerator / denominator)

            z = ((d1 + d2)/2)*np.tan(phi)
            phi_a += phi
            if (z < 4):
                x_fw += z
                count += 1
            if count >= 10:
                phi_a /= count
                x_fw /= count
                print(d1, d2, f"Phi: {np.degrees(phi_a):.2f}, Forward_X: {x_fw:.3f}")
                if ((x_fw < 0.15 and ((gates[0][2]+gates[1][2])/2) <= 1800)):
                    state = "GOAL"
                    clear_goal = True
                else:
                    state = "MOVE"
                count = 0
                continue


        if state == "MOVE" and not turtle.is_shutting_down():
            turtle.cmd_velocity(angular = 0, linear = 0)
            cv2.imshow("result", result)

            turn_deg(turtle, 90, side)
            go_fw(turtle, x_fw, side)

            x_fw = 0
            phi_a = 0
            ball_found = False
            dist_between_poles = 0.7
            state = "ROTATE"

        if state == "ROTATE" and not turtle.is_shutting_down():
            turtle.cmd_velocity(angular = 0, linear = 0)
            complete_rotation = True

            if (len(gates) == 2):
                result = cv2.circle(result, ((int((gates[1][1] + gates[0][1]) / 2)), 240), 2, (255, 0, 30), 5)

                if ((-5 < CENTER_IMG[1] - (int((gates[1][1] + gates[0][1]) / 2) < 5) and (gates[1][2]*0.97 < gates[0][2] < gates[1][2]*1.03)) and ball is None):
                    state = "GOAL"
                    clear_goal = True
                    continue

            cv2.imshow("result", result)

            if ball is None:
                state = "LOOK"
                continue
    
            # to big distance to gate center -> skip
            if (((CENTER_IMG[1] - ((gates[0][1]+ gates[1][1])/2)) > 55) and ((CENTER_IMG[1] - ball[0]) < 25)) or ignore_ball == True:
                state = "FINDGATE"
                complete_rotation = False
                ignore_ball = False
                continue

            PARAM = max(int((4.5 * ((gates[0][2]/1000+gates[1][2]/1000)/2) ** 2)), 5) 
            print(PARAM, (CENTER_IMG[1] - ball[0]), (CENTER_IMG[1] - ((gates[0][1]+ gates[1][1])/2)))

            if ((CENTER_IMG[1] - ball[0])) < -7:
                turtle.cmd_velocity(angular = -0.2 - abs(CENTER_IMG[1] - ball[0]) * 0.003)
            elif CENTER_IMG[1] - ball[0] > 7:
                turtle.cmd_velocity(angular = 0.2 + abs(CENTER_IMG[1] - ball[0]) * 0.003)
            else:
                if ((CENTER_IMG[1] - ball[0]) > -PARAM and (CENTER_IMG[1] - ((gates[0][1]+ gates[1][1])/2)) > -PARAM) and ((CENTER_IMG[1] - ball[0]) < PARAM and (CENTER_IMG[1] - ((gates[0][1]+ gates[1][1])/2)) < PARAM):
                    state = "GOAL"
                else:
                    complete_rotation = False
                    state = "FINDGATE"
                turtle.cmd_velocity(angular = 0, linear = 0)


        if state == "GOAL" and not turtle.is_shutting_down():
            turtle.cmd_velocity(angular = 0, linear = 0)

            if ((ball is not None and len(gates) == 2) or clear_goal == True):
                if clear_goal == True:
                    PARAM = 0.6
                else:
                    PARAM = (ball[2]/1000) * 2

                t3 = get_time()
                while (get_time() - t3 < PARAM):
                    turtle.cmd_velocity(linear = 1)
                state = "DEATH"
            else:
                state = "ROTATE"



def exit_thread(turtle):
    while not turtle.is_shutting_down():
        cv2.waitKey(1)
        if BUMPER:
            print('\nEND: Bumper exit')
            turtle.play_sound(sound_id=4)
            break
        if turtle.is_shutting_down():
            print('\nEND: Ctrl + C exit')
            turtle.play_sound(sound_id=6)
            break

def main():
    turtle = Turtlebot(pc=False, rgb=True, depth=True)
    turtle.register_bumper_event_cb(bumper_callback)

    turtle.play_sound(sound_id=1)

    main_thr = threading.Thread(target=main_thread, args=(turtle,))
    exit_thr = threading.Thread(target=exit_thread, args=(turtle,))

    main_thr.start()
    exit_thr.start()

    exit_thr.join()
    main_thr.join()

if __name__ == '__main__':
    main()

