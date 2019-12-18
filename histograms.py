import os
import cv2 as cv

ORIGINAL_HEIGHT = 576
ORIGINAL_WIDTH = 768
image_folder_path = 'Video/img1/'

def computeHistogram(frame_i, bbox):
    '''
    :param frame_i: index of frame (1 = first)
    :param bbox: in format: [frame, left, top, width, height, x_center, y_center]
    :return:
    '''
    # get correct image
    file = os.listdir(image_folder_path)[int(frame_i - 1)]
    image = cv.imread(image_folder_path + file)

    # crop image to bounding box size
    y = int(bbox['top'])
    y_min = 0 if y < 0 else y  # check that bbox is not outside image frame
    x = int(bbox['left'])
    x_min = 0 if x < 0 else x  # check that bbox is not outside image frame
    h = int(bbox['height'])
    w = int(bbox['width'])
    y_max = y_min + h
    x_max = x_min + w
    # y_max = ORIGINAL_HEIGHT if y_max > ORIGINAL_HEIGHT else y_max
    # x_max = ORIGIN_WIDTH if x_max > ORIGIN_WIDTH else x_max
    crop_img = image[y_min:y_max, x_min:x_max]
    # cv.imshow("crop", crop_img)
    # extract histogram and return
    crop_img = cv.split(crop_img)
    return [
       cv.calcHist([crop_img[0]], [0], None, [50], [0, 256]),
       cv.calcHist([crop_img[1]], [0], None, [50], [0, 256]),
       cv.calcHist([crop_img[2]], [0], None, [50], [0, 256])
    ]

def compareHists(hist_a, hist_b):
    r_chan = cv.compareHist(hist_a[0], hist_b[0], 0)
    g_chan = cv.compareHist(hist_a[1], hist_b[1], 0)
    b_chan = cv.compareHist(hist_a[2], hist_b[2], 0)
    return (r_chan + g_chan + b_chan) / 3