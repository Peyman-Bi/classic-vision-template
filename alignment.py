import cv2
import numpy as np
import os


def distance(p1, p2):
    # method to sort corners
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def order_corner_points(corners):
    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0], corner[1]) for corner in corners]

    # Sort corners to adjust their orders using distance measure
    corners = sorted(corners, key=lambda p: distance(p, corners[0]))

    # Permute top right and top left of the image based on image angle
    top_r, top_l = corners[0], corners[2]
    if top_r[0] < top_l[0]:
        top_l, top_r, bottom_r, bottom_l = corners[0], corners[2], corners[3], corners[1]
    else:
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[2], corners[3], corners[1]
    return top_l, top_r, bottom_r, bottom_l


def perspective_transform(image, corners, border):
    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[border, border], [width-border, border], [width-border, height-border],
                    [border, height-border]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def rotate_image(image, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def align_image(image, blur_filter, canny_args, border):

    # Convert image to gray scale
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth image with the filter
    gray_blured = cv2.bilateralFilter(image, *blur_filter)

    # detect edges
    edged = cv2.Canny(gray_blured, *canny_args)

    # apply closing morphology transformation to close small holes inside the foreground objects
    kernel = np.ones((2, 2),np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # detect contours
    contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Sort contours to pick just larger ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screen_cnt = None
    # loop over our contours
    for contour in contours:
        # approximate the contour by a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # check to find the first rectangle
        if len(approx) == 4:
            screen_cnt = approx.squeeze(axis=1)
            transformed = perspective_transform(original_image, screen_cnt, border)
            break

    one_detected = False
    # Draw contours
    if screen_cnt is not None:
        cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 3)
        rotated = rotate_image(transformed, 0)
        one_detected = True
    else:
        print('No rectangular contour detected!')

    if one_detected:
        return rotated

    return image


if __name__ == '__main__':
    results_path = './part1_results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # blur filter
    blur_agrs = (3, 15, 15)
    # canny edge detection args
    canny_args = (20, 300)
    # border to cut picture
    border = 30
    images_path = './interview_task_ComputerVision/FirstPart/'

    for file in os.listdir(images_path):
        save_path = os.path.join(results_path, file)
        file_path = os.path.join(images_path, file)
        print(f'Processing image {file}')
        image = cv2.imread(file_path)
        output_image = align_image(image, blur_agrs, canny_args, border)
        cv2.imshow("Aligned Image", output_image)
        cv2.waitKey(0)
        cv2.imwrite(save_path, output_image)

    cv2.destroyAllWindows()


