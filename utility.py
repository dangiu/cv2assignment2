import numpy as np
import cv2 as cv
import random as rng
import os
import pandas as pd

# parameters
image_folder_path = 'Video/img1/'

def parseTracks(tracks):
    frame_n = []
    id = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    for t in tracks:
        for bbox in t['bboxs']:
            #frame, id, minx, miny, maxx, maxy
            frame_n.append(bbox['frame'])
            id.append(tracks.index(t))
            minx.append(bbox['left'])
            miny.append(bbox['top'])
            maxx.append(bbox['left'] + bbox['width'])
            maxy.append(bbox['top'] + bbox['height'])
    d = {'frame': frame_n, 'id': id, 'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
    df = pd.DataFrame(data=d)
    return df

def getRectByFrameAndParsedTracks(frame_index, parsedTracks):
    return parsedTracks.loc[parsedTracks['frame'] == frame_index]


def showTracksOnImage(parsedTracks):
    # generate color for each id
    ids = parsedTracks.id.unique()
    colors = {}
    for i in ids:
        colors[i] = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

    image_paths = []
    frameIndex = 1
    for file in os.listdir(image_folder_path):
        image_paths.append((frameIndex, image_folder_path + file))
        frameIndex += 1

    for frame_i, frame_path in image_paths:
        frame = cv.imread(frame_path)
        if frame is None:
            print('Could not open or find the image: ' + frame_path)
            exit(0)
        rows = getRectByFrameAndParsedTracks(frame_i, parsedTracks)
        for index, row in rows.iterrows():
            src = cv.rectangle(frame, (int(row['minx']), int(row['miny'])), (int(row['maxx']), int(row['maxy'])), colors[int(row['id'])], 2)

        cv.imshow('frame', frame)
        cv.waitKey(10)

def outputDetections():
    """
    Create description.txt for delivery
    :return:
    """
    input_file = 'bb_out.csv'
    names = ['frame', 'bbox', 'left', 'top', 'width', 'height']
    bb_data = pd.read_csv(input_file, header=None, names=names)
    # pre-process bounding box data
    bb_data['frame'] = bb_data['frame'] + 1  # shift frames to align with ground truth data
    bb_data = bb_data.drop(['bbox'], 1)
    bb_data.to_csv('detection.txt', header=False, index=False)

def outputTracks(tracks):
    """
        Create tracking.txt for delivery by passing computed tracks (unparsed)
        :return:
    """
    # parse tracks
    frame_n = []
    id = []
    x_center = []
    y_center = []
    for t in tracks:
        for bbox in t['bboxs']:
            # frame, id, minx, miny, maxx, maxy
            frame_n.append(bbox['frame'])
            id.append(tracks.index(t))
            x_center.append(bbox['left'] + (bbox['width'] / 2))
            y_center.append(bbox['top'] + (bbox['height'] / 2))
    d = {'frame': frame_n, 'id': id, 'x_center': x_center, 'y_center': y_center}
    df = pd.DataFrame(data=d)
    df = df.sort_values(by=['frame'])
    # output to csv
    df.to_csv('tracking.txt', header=False, index=False)

if __name__ == '__main__':
    outputDetections()