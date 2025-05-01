import cv2
import numpy as np

#To find the center of the colored object on the X-Y frame
def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        return None

#
cap = cv2.VideoCapture(0)
#this dictionray is used to add colors with their HSV values
color_ranges = {
    'red': [(np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    'green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
    'blue': [(np.array([100, 150, 0]), np.array([140, 255, 255]))]
}

#Another dictionary but this is just used to have BGR values to use them for the dots on the centers
color_map = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}

while True:
    #Splitting the captured live video into frames to process them
    ret, frame = cap.read()

    #then converting the frame from BGR which is the defult for cv2 to HSV to be able to mask the colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Processing each color range
    for color_name, ranges in color_ranges.items():
        largest_contour = None
        largest_area = 0

        for lower, upper in ranges:
            #Creating a mask for the current color range
            mask = cv2.inRange(hsv, lower, upper)

            #then drawing a contour around that mask to be able to clculate it's area and find it's center
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #making sure to focus on the largest object with that color
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    largest_contour = contour

        #drawing and printing its center
        if largest_contour is not None:
            center = get_center(largest_contour)
            if center:
                #using the dictionary of BGR above to draw on the window
                cv2.circle(frame, center, 5, color_map[color_name], -1)
                #getting the coordinates of the color to be sent to the robotic arm
                print(f"Detected largest {color_name} at coordinates: {center}")

    cv2.imshow('Frame', frame)


    if cv2.waitKey(1) & 0xFF == 27:  #27= esc button to close the window
        break

#closing the window and ending the code if the loop was broken
cap.release()
cv2.destroyAllWindows()