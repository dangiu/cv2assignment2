from matplotlib import pyplot as plt
from matplotlib import patches as patches
import gluoncv
from gluoncv import model_zoo, data, utils
import numpy as np
from PIL import Image
import csv
import os

# Configuration parameters
# original dimensions
ORIGINAL_HEIGHT = 576
ORIGINAL_WIDTH = 768
# imported dimensions
HEIGHT = 600
WIDTH = 800


def extractClassData(bb_ids, scores, bbs, selectedClass, minConfidence=0.5):
    classId = network.classes.index(selectedClass)
    shape = bb_ids.shape
    nImages = shape[0]
    nBoxes = shape[1]

    #save indexes wanted
    imIndexes = []
    boxIndexes = []

    for image in range(nImages):
        for box in range(nBoxes):
            if ((bb_ids[image][box][0] == classId) and (scores[image][box][0] >= minConfidence)):
                if(not image in imIndexes):
                    imIndexes.append(image)
                boxIndexes.append(box)

    return bb_ids[imIndexes, boxIndexes], scores[imIndexes, boxIndexes], bbs[imIndexes, boxIndexes]



if __name__ == '__main__':

    #image paths
    im_paths = []

    folder_path = 'Video/img1/'
    for file in os.listdir(folder_path):
        im_paths.append(folder_path + file)

    im_path = 'Video/img1/000001.jpg'

    #import Faster R-CNN pretrained on Pascal VOC with ResNet-50 as backbone
    network = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    x_list, orig_img_list = data.transforms.presets.rcnn.load_test(im_paths)

    #output file
    with open('bb_out.csv', mode='w') as output:
        out_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #frame counter
        frameCounter = 0;

        for x in x_list:
            box_ids, scores, bboxes = network(x)
            p_box_ids, p_scores, p_bboxes = extractClassData(box_ids, scores, bboxes, 'person', minConfidence=0.5)
            new_bboxes = gluoncv.data.transforms.bbox.resize(p_bboxes, (WIDTH, HEIGHT), (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
            #bbox counter
            bboxCounter = 0
            for bbox in new_bboxes:
                rect_w = bbox[2] - bbox[0]
                rect_w = rect_w.asscalar() #width
                rect_h = bbox[3] - bbox[1]
                rect_h = rect_h.asscalar() #height
                x_coord = bbox[0]
                x_coord = x_coord.asscalar() #left
                y_coord = bbox[1]
                y_coord = y_coord.asscalar() #bottom
                #write to file
                #frameN, bboxN, left, top, width, height
                out_writer.writerow([frameCounter, bboxCounter, x_coord, y_coord, rect_w, rect_h])
                bboxCounter += 1
            frameCounter += 1