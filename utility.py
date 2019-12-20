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

def showTracksAndBoxes(tracks):
    # t = {'started_on': frame, 'last_updated_on': frame, 'bboxs': [d]}

    # generate color for each track
    for t in tracks:
        t['color'] = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

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
        for t in tracks:
            if t['started_on'] <= frame_i and t['last_updated_on'] >= frame_i:   # check if track is active
                # display segment for each 2 bb + bb of current frame (if exist)
                prevBB = None
                for b in t['bboxs']:
                    if b['frame'] <= frame_i:
                        if b['frame'] == frame_i:
                            # display bbox
                            #frame, left, top, width, height, x_center, y_center, hist
                            minx = int(b['left'])
                            miny = int(b['top'])
                            maxx = int(b['left'] + b['width'])
                            maxy = int(b['top'] + b['height'])
                            cv.rectangle(frame, (minx, miny), (maxx, maxy), t['color'], 2)
                        if prevBB is not None:
                            # display segment between prevBB and b
                            cv.line(frame, (int(prevBB['x_center']), int(prevBB['y_center'])), (int(b['x_center']), int(b['y_center'])), t['color'], thickness=2, lineType=8, shift=0)
                        prevBB = b
                    else:
                        break
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

def plotDetection(frame_index, bb_file_name):
    file_index = frame_index - 1
    file = os.listdir(image_folder_path)[file_index]
    names = ['frame', 'bbox', 'left', 'top', 'width', 'height']
    bb_data = pd.read_csv(bb_file_name, header=None, names=names)
    # extract data of relevant frame
    bb_data = bb_data.loc[bb_data['frame'] == frame_index]
    # pre-process bounding box data
    bb_data['frame'] = bb_data['frame'] + 1  # shift frames to align with ground truth data
    bb_data['x_center'] = bb_data['left'] + (bb_data['width'] / 2)  # compute x of bb centroid
    bb_data['y_center'] = bb_data['top'] + (bb_data['height'] / 2)  # compute y of bb centroid
    bb_data['right'] = bb_data['left'] + bb_data['width']
    bb_data['bottom'] = bb_data['top'] + bb_data['height']
    # read frame
    frame = cv.imread(image_folder_path + "/" + file)
    # plot bb onto frame
    for index, row in bb_data.iterrows():
        src = cv.rectangle(frame, (int(row['left']), int(row['top'])), (int(row['right']), int(row['bottom'])),
                           (0, 0, 255), 2)
    cv.imshow('frame', frame)
    cv.waitKey(0)

if __name__ == '__main__':
    outputDetections()
    #plotDetection(20, 'bb_out.csv')