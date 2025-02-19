'''
在cube_easy的基础上修改了如下：
1. canforward函数新增pixelCount判断
2. 将速度划分为两个阶段，防止两电机起步速度不同的问题
3. 在getthrough和findtarget中间穿插函数turnBack
'''

import threading
import numpy as np
import cv2
from collections import deque
import RPi.GPIO as GPIO
import time
import wiringpi as wpi
import cube_recognizer.libRecognize as lR
from collections import OrderedDict

def sign(x):
    if x > 0:
        return 1.0
    else:
        return -1.0
    
def getDistance():
    global distance
    try:
        address = 0x74 #i2c device address
        h = wpi.wiringPiI2CSetup(address) #open device at address
        wr_cmd = 0xb0  #range 0-5m, return distance(mm)
        while True:
            wpi.wiringPiI2CWriteReg8(h, 0x2, wr_cmd)
            wpi.delay(33) #unit:ms  MIN ~ 33
            HighByte = wpi.wiringPiI2CReadReg8(h, 0x2)
            LowByte = wpi.wiringPiI2CReadReg8(h, 0x3)
            Dist = (HighByte << 8) + LowByte
            distance = Dist/10.0
            # print('Distance:', distance, 'cm')
    except KeyboardInterrupt:
        pass
        print("END!!!")

def ValidSpeed(speed):
    if speed > 100:
        return 100
    elif speed < 0:
        return 0
    return speed

# 遇见魔方转弯
def turnLeft():
    pwmRight.ChangeDutyCycle(rightSpeed * 0.8)
    pwmLeft.ChangeDutyCycle(0)
def turnRight():
    pwmRight.ChangeDutyCycle(0)
    pwmLeft.ChangeDutyCycle(leftSpeed * 0.8)
    
# find target
def turnLeftFindTarget():
    pwmRight.ChangeDutyCycle(30)
    pwmLeft.ChangeDutyCycle(5)
def turnRightFindTarget():
    pwmRight.ChangeDutyCycle(5)
    pwmLeft.ChangeDutyCycle(30)

# 直行调整角度
def turnStraightLeft():
    pwmRight.ChangeDutyCycle(rightSpeed)
    pwmLeft.ChangeDutyCycle(ValidSpeed(leftSpeed + adjust[2] * 0.5))
def turnStraightRight():
    pwmRight.ChangeDutyCycle(ValidSpeed(rightSpeed - adjust[2] * 0.5))
    pwmLeft.ChangeDutyCycle(leftSpeed)

def turnLeftTurnBack():
    pwmRight.ChangeDutyCycle(50)
    pwmLeft.ChangeDutyCycle(15)

def turnRightTurnBack():
    pwmRight.ChangeDutyCycle(15)
    pwmLeft.ChangeDutyCycle(50)

def moveForward():
    pwmRight.ChangeDutyCycle(rightSpeed)
    pwmLeft.ChangeDutyCycle(leftSpeed)

def stop():
    pwmRight.ChangeDutyCycle(0)
    pwmLeft.ChangeDutyCycle(0)

def getCornerXCor(image):
    nowColor, mask, res, _ = getColor(image)

    cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(cnts) > 0:  # If contours are found
        c = max(cnts, key=cv2.contourArea)  # Find the largest contour by area
        ((x, y), radius) = cv2.minEnclosingCircle(c)  # Find the minimum enclosing circle

        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 5:
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)

        # Get the specified corner point
        corner_point = get_corner_from_contour(c, nowTurnDirection)
        cv2.circle(img, (corner_point[0], center[1]), 5, (255, 0, 0), -1)

        # Print the x-coordinate of the corner point
        print(f"X-coordinate of the corner point: {corner_point[0]}")
        xC = corner_point[0]
        return xC
    else:
        return None

def getColor(image):
    global Colors
    image = image[:, :-50]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert captured frame from RGB to HSV

    kernel = np.ones((3, 3), np.uint8)
    target_mask = None
    target_res = None

    pixel_count = -1
    for color in Colors:
        lower = lowerRange[color]
        upper = upperRange[color]

        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=5)
        res = cv2.bitwise_and(image, image, mask=mask)
        if pixel_count < cv2.countNonZero(mask):
            pixel_count = cv2.countNonZero(mask)
            target_mask = mask
            target_res = res
            nowColor = color
    
    cv2.imshow("img", image)
    cv2.imshow("mask", target_mask)
    cv2.imshow("res", target_res)

    print(f"getColor()中的pixelCount:{pixel_count}")

    return nowColor, target_mask, target_res, pixel_count


def get_corner_from_contour(contour, corner):
    """
    Get the specified corner (left_bottom or right_bottom) from the contour.
    """
    if corner == LEFT:
        # 获取最底部的点
        bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
        # 获取最底部点的最左侧的点
        point = min([pt[0] for pt in contour if pt[0][1] == bottom_point[1]], key=lambda x: x[0])
    elif corner == RIGHT:
        # 获取最底部的点
        bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
        # 获取最底部点的最右侧的点
        point = max([pt[0] for pt in contour if pt[0][1] == bottom_point[1]], key=lambda x: x[0])
    else:
        raise ValueError("Corner must be 'left_bottom' or 'right_bottom'")
    
    return point

def getDiff() :
    global diff
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            nowColor, mask, res, _ = getColor(img)

            cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            center = None

            if len(cnts) > 0:  # If contours are found
                c = max(cnts, key=cv2.contourArea)  # Find the largest contour by area
                ((x, y), radius) = cv2.minEnclosingCircle(c)  # Find the minimum enclosing circle

                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 5:
                    cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(img, center, 5, (0, 0, 255), -1)

                direction = Color2Direction[nowColor]
                # Get the specified corner point
                corner_point = get_corner_from_contour(c, direction)
                cv2.circle(img, (corner_point[0], center[1]), 5, (255, 0, 0), -1)
                
                # Print the x-coordinate of the corner point
                print(f"X-coordinate of the corner point: {corner_point[0]}")

                xCor = corner_point[0]
                diff = xCor - 320
                print("The difference：", diff)

            k = cv2.waitKey(30) & 0xFF
            if k == 32:  # Press space to exit
                break
    except KeyboardInterrupt:
        print("END!!!")
        pass
    
def GetThrough(nowTurnDirection, turnCount):
    # Turn to a safe direction
    print("getThrough: ", nowTurnDirection)
    if nowTurnDirection == LEFT:
        print("isTurnLeft!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Distance:",distance)
        print("pixelCount:",pixelCount)
        turnLeft()
        time.sleep(0.55 - turnCount * 0.0)
    elif nowTurnDirection == RIGHT:
        print("isTurnRight!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Distance:",distance)
        print("pixelCount:",pixelCount)
        turnRight()
        time.sleep(0.55 - turnCount * 0.0)
    # Move forward
    # moveForward()
    # print("isMovingForward###########################")
    # time.sleep(0.05)
    stop()
    print("stop############################")

def findTarget(image, pixelCount):
    while pixelCount < pixelMin:
        print('Distance:', distance, 'cm')
        print('PixelCount:',pixelCount)
        _, _, _, pixelCount= getColor(img)
        if nowTurnDirection == LEFT:
            turnRightFindTarget()
            print("isTurningRightFindTarget @@@@@@@@@@@@@@@@@@@@@@")
        elif nowTurnDirection == RIGHT:
            turnLeftFindTarget()
            print("isTurningLeftFindTarget @@@@@@@@@@@@@@@@@@@@@@")
    print("finishFinding @@@@@@@@@@@@@@@@@@@@@@@@@")

def CanForward():
    return distance > 85

def turnBack(nowTurnDirection):
    print("Start Turning Back &&&&&&&&&&&&&&&&&&")
    if nowTurnDirection == LEFT:
        turnRightTurnBack()
        time.sleep(0.5)
        # stop()
        # time.sleep(1)
    elif nowTurnDirection == RIGHT:
        turnLeftTurnBack()
        time.sleep(0.5)
        # stop()
        # time.sleep(1)
    print("Finish Turning Back &&&&&&&&&&&&&&&&")
    moveForward()
    time.sleep(0.5)
    stop()
    time.sleep(1)

'''
-------------------------------START OF MAIN---------------------------------------
'''
# Video capture from the default camera
cap = cv2.VideoCapture(0)

# Queue to store the points
pts = deque(maxlen=128)

EA, I2, I1, EB, I4, I3 = (13, 19, 26, 16, 20, 21)
FREQUENCY = 55
GPIO.setmode(GPIO.BCM)
GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
GPIO.output([EA, I2, EB, I3], GPIO.LOW)
GPIO.output([I1, I4], GPIO.HIGH)
pwmRight = GPIO.PWM(EA, FREQUENCY)
pwmLeft = GPIO.PWM(EB, FREQUENCY)
pwmRight.start(0)
pwmLeft.start(0)

error = [0.0] * 3
adjust = [0.0] * 3

kp = 1.8
ki = 0.2
kd = 0.03
imageCenter = 320
stageCutter = 20
leftSpeed_st1 = 75
rightSpeed_st1 = 65
leftSpeed_st2 = 65
rightSpeed_st2 = 65
control = 45

LEFT = 0
RIGHT = 1

# Define the HSV color range
lower_blue = np.array([100, 40, 40])
upper_blue = np.array([140, 255, 255])

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

lower_yellow = np.array([25, 30, 150])
upper_yellow = np.array([35, 255, 255])

lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

lower_orange = np.array([5, 120, 100])
upper_orange = np.array([15, 255, 255])

lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])

lowerRange = {"red": lower_red, "green": lower_green, "blue": lower_blue, "yellow": lower_yellow, "orange": lower_orange}
upperRange = {"red": upper_red, "green": upper_green, "blue": upper_blue, "yellow": upper_yellow, "orange": upper_orange}


Color2Direction = {"red":RIGHT, "blue":RIGHT, "orange":LEFT, "green":LEFT}
Colors = ["red", "blue", "green", "orange"]

print("ready")
input()

out = cv2.VideoWriter("movie.avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 17, (640, 480))  # 打开/新建视频文件用于写入,帧率=17,帧尺寸=640x480
font = cv2.FONT_HERSHEY_SIMPLEX

diff = 0
distance = 0

# 设置find target的最小像素面积
pixelMax = 20000
pixelMin = 4500 
pixelCount = 0

thread1 = threading.Thread(target = getDistance)
thread1.start()

thread2 = threading.Thread(target = getDiff)
thread2.start()
time.sleep(3)

global img
global turnMode
turnMode = 0

try:
    while True:
        ret, img = cap.read()
        nowColor, mask, res, pixelCount = getColor(img)

        if stageCutter > 0:
            leftSpeed = leftSpeed_st1
            rightSpeed = rightSpeed_st1
            stageCutter -= 1
        else:
            leftSpeed = leftSpeed_st2
            rightSpeed = rightSpeed_st2

        # updates the pid
        error[0] = error[1]
        error[1] = error[2]
        error[2] = diff

        adjust[0] = adjust[1]
        adjust[1] = adjust[2]
        adjust[2] = adjust[1] + kp * (error[2] - error[1]) + ki * error[2] + kd * (error[2] - 2 * error[1] + error[0])
        print(f"adjust:{adjust[2]}")

        if abs(adjust[2]) > control:
            adjust[2] = sign(adjust[2]) * control

        cv2.putText(img, "turnCount:"+str(), (10, 30), font, 1, (0, 255, 0), 2)

        if distance > 85:
            turnMode = 0
        else:
            turnMode = 1

        if CanForward():
            # Safe. Keep moving.
            # Turn right
            if adjust[2] > 20:
                lastAct = turnStraightRight()
                print("isTurnStraightRight!!!!!!!!!!!!!!!!!!!!!!!")
            # Turn left
            elif adjust[2] < -20:
                lastAct = turnStraightLeft()
                print("isTurnStraightRight!!!!!!!!!!!!!!!!!!!!!!!")
            # Go forward
            else:
                print("isForward!!!!!!!!!!!!!!!!!!!!!!!!!!")
                moveForward()
        else:
            # # Too close. Get through the obstacle.
            print("notCanForward!!!!!!!!!!!!!!!!!!!!!!!!!")

            turnCount = 0

            ret, img = cap.read()
            nowColor, _, _, pixelCount= getColor(img)          
            nowTurnDirection = Color2Direction[nowColor]

            print(f"nowTurnDirection:{nowTurnDirection}")
            GetThrough(nowTurnDirection, turnCount)
            # stop()
            # time.sleep(1)

            turnBack(nowTurnDirection)
            
            # findTarget(img, pixelCount)
            print("********************Start Finding Target********************")

            _, _, _, pixelCount= getColor(img)

            while pixelCount < pixelMin and pixelCount > pixelMax:

                turnCount += 1
                
                print('Distance:', distance, 'cm')
                ret, img = cap.read()
                nowColor, _, _, pixelCount= getColor(img)  
                print('pixelCount:',pixelCount)
                print("nowColor:",nowColor)
                if nowTurnDirection == LEFT:
                    turnRightFindTarget()
                    print("isTurningRightFindTarget@@@@@@@@@@@@@@@@@@@@@@")
                elif nowTurnDirection == RIGHT:
                    turnLeftFindTarget()
                    print("isTurningLeftFindTarget@@@@@@@@@@@@@@@@@@@@@@")

            print("finishFinding@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("turnCount:",turnCount)



            # Updates the nowColor
            
            # 下面一段不明意义！！！
            if nowColor is None:
                nowColor = Colors[0]
                stop()
                time.sleep(1)
                # Finds the new target
                findTarget(img)
                nowTurnDirection = Color2Direction[nowColor]

        out.write(img)
except KeyboardInterrupt:
    stop()
    pass
    print("END!!!")


stop()
out.release()
cap.release()
cv2.destroyAllWindows()
pwmRight.stop()
pwmLeft.stop()
thread1.join()
thread2.join()
GPIO.cleanup()