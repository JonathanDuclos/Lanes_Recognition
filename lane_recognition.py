import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])

#Make a average between the lines x and y coordinates, mescling
def average_slope_intercepts(image, lines):
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

#Draw lines in a image
def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image

#Creating a canny image
def create_canny_image(image):
    #Adding a grayscaling on image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Adding the gausian Blur and Treshhold on image
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    return canny_image

#Creating a Region of Interesting mask
def create_region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Importing image and videos
video = cv2.VideoCapture('test2.mp4')

#Using video
while(video.isOpened()):
    _, frame = video.read()
    canny = create_canny_image(frame)

    #Bitwise_and (making a AND with the returned mask from intereting region and our image)
    cropped_image = create_region_of_interest(canny)

    #Hough Lines and draw them
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercepts(frame, lines)
    drawed_lines_image = draw_lines(frame, average_lines)

    #Misc both images: the lines with the original image
    combo_image = cv2.addWeighted(frame, 0.8, drawed_lines_image, 1, gamma=1)

    #Show up the image
    cv2.imshow('Finding Lanes - Jonathan Duclos', combo_image)
    cv2.waitKey(1)