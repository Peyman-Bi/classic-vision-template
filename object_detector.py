import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def find_circulars(img, blur_args, hough_kwargs):

    # convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth using specified kernel.
    gray_blurred = cv2.bilateralFilter(gray, *blur_args)

    # apply Hough transform to detect circles
    detected_circles = cv2.HoughCircles(gray_blurred, **hough_kwargs)

    # draw detected circles
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

    return img, detected_circles.shape[1]


if __name__ == '__main__':

    results_path = './part2_results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    """
    Processing images: 1, 2, 4, 5
    """

    # define Hough algorithm arguments for each image
    hough_dict = dict(
        img1=dict(
            method=cv2.HOUGH_GRADIENT,
            dp=1, minDist=100,
            param1=210, param2=25,
            minRadius=40, maxRadius=200
        ),
        img2=dict(
            method=cv2.HOUGH_GRADIENT,
            dp=1, minDist=50,
            param1=200, param2=30,
            minRadius=20, maxRadius=40
        ),
        img4=dict(
            method=cv2.HOUGH_GRADIENT,
            dp=1, minDist=100,
            param1=200, param2=35,
            minRadius=40, maxRadius=200
        ),
        img5=dict(
            method=cv2.HOUGH_GRADIENT,
            dp=1, minDist=100,
            param1=200, param2=35,
            minRadius=40, maxRadius=170
        )
    )

    # define smoothing filter arguments for each image
    blur_dict = dict(
        img1=(7, 150, 150),
        img2=(4, 20, 20),
        img4=(5, 70, 70),
        img5=(4, 60, 60)
    )

    image_names = dict(
        img1='1.jpg',
        img2='2.jpg',
        img4='4.jpg',
        img5='5.jpg'
    )

    for key in image_names.keys():
        image_name = image_names[key]
        img = cv2.imread('interview_task_ComputerVision/SecondPart/'+image_name)
        blur_args = blur_dict[key]
        hough_kwargs = hough_dict[key]
        objects_img, num_objects = find_circulars(img, blur_args, hough_kwargs)
        print(f'Number of detected objects in image {image_name}: {num_objects}')
        cv2.imshow("Detected Objects", objects_img)
        cv2.waitKey(0)
        cv2.imwrite(results_path+image_name, objects_img)

    """
    Processing images: 3
    """

    img = cv2.imread('interview_task_ComputerVision/SecondPart/3.jpg')
    original_image = img.copy()

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask of green (28, 25, 25), (70, 255,255)
    mask = cv2.inRange(hsv, (28, 25, 25), (70, 255,255))

    ## slice the green
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    # mask green from background
    img[imask] = 0

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect edges from the rest of image
    edged = cv2.Canny(gray, 25, 200)

    # apply closing morphology transformation to close small holes inside the foreground objects
    kernel = np.ones((3, 3),np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # detect contours
    contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # filter contours by area size to omit wrong detected objects
    contours = list(filter(lambda x: cv2.contourArea(x) > 200, contours))

    # draw contours on original image
    cv2.drawContours(original_image, contours, -1, (0, 0, 255), 2)

    print(f'Number of detected objects in image 3.jpg: {len(contours)}')
    cv2.imshow("Detected Objects", original_image)
    cv2.waitKey(0)
    cv2.imwrite(results_path+'3.jpg', original_image)

    """
    Processing images: 6
    """

    img = cv2.imread('interview_task_ComputerVision/SecondPart/6.jpg')

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth using specified kernel.
    gray_blurred = cv2.bilateralFilter(gray, 4, 200, 200)

    # detect edges and close holes
    edged = cv2.Canny(gray_blurred, 70, 300)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # detect contours
    contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # draw contours
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    print(f'Number of detected objects in image 6.jpg: {len(contours)}')
    cv2.imshow("Detected Circle", img)
    cv2.waitKey(0)
    cv2.imwrite(results_path + '6.jpg', img)
    cv2.destroyAllWindows()






