import pandas as pd
import numpy as np
import gluoncv
import math
import plotUtil
import histograms as hist
import os
import cv2 as cv

# parameters
input_file = 'bb_out.csv'
names = ['frame', 'bbox', 'left', 'top', 'width', 'height']
max_distance_thresh = 200
track_age_thresh = 16
min_track_len = 14
use_color_histogram = True

image_folder_path = 'Video/img1/'

output_name = 'tracking.csv'

def getDetectionsByFrame(detections, frame_index):
    """
    :param detections: dataframe containing detections data
    :param frame_index: frame for which the detection are wanted
    :return: list of detection, each a dictionary form:
            frame, left, top, width, height, x_center, y_center, hist
    """
    df = detections.loc[detections['frame'] == frame_index]
    return df.drop(['bbox'], 1).to_dict('records')

def getFrameList(detections):
    return detections.frame.unique()

def computePointsDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def computeDistance(track, detection, frame_i):
    last_track_detect = track['bboxs'][-1]
    # d = computePointsDistance(last_track_detect[-2], last_track_detect[-1], detection[-2], detection[-1])
    d = computePointsDistance(last_track_detect['x_center'], last_track_detect['y_center'], detection['x_center'], detection['y_center'])

    if(use_color_histogram):
        hist_track = track['hist']
        hist_detect = detection['hist']
        hist_d = 1 - hist.compareHists(hist_track, hist_detect)
        # normalize d and hist_d to make them comparable

        total_d = d * 0.25 + hist_d * 0.75
    else:
        total_d = d
    return total_d

def computeBestMatch(tracks, detections, frame_i):
    best_distance = float('inf')
    best_track = None
    best_detection = None
    for t in tracks:
        for d in detections:
            distance = computeDistance(t, d, frame_i)
            if distance < best_distance:
                best_distance = distance
                best_track = t
                best_detection = d
    return best_distance, best_track, best_detection

def track(detections):
    tracks = []
    stable_tracks = []
    for frame in getFrameList(detections):
        print(frame)
        updated_tracks = []
        frame_detections = getDetectionsByFrame(detections, frame)
        # matching part
        no_match = False
        while len(frame_detections) > 0 and not no_match:
            distance, t, d = computeBestMatch(tracks, frame_detections, frame)
            #print('Best: ' + str(score))
            if distance > max_distance_thresh:
                no_match = True
            else:
                # add detection to track (best match pair)
                t['bboxs'].append(d)
                t['last_updated_on'] = frame
                updated_tracks.append(t)
                # remove t and d from available tracks and detection
                del tracks[tracks.index(t)]
                del frame_detections[frame_detections.index(d)]
        # handle unmatched detections (if any remains)
        new_tracks = []
        for d in frame_detections:
            # try merge with existing track by means of reid
            # if they are in the center of the image is more probable that they belong to already tracked object
            # TODO IMPLEMENT
            # otherwise create new track
            if use_color_histogram:
                t = {'started_on': frame, 'last_updated_on': frame, 'bboxs': [d], 'hist': d['hist']}
            else:
                t = {'started_on': frame, 'last_updated_on': frame, 'bboxs': [d]}
            new_tracks.append(t)
        # check for stable tracks
        for t in tracks:
            if (frame - t['last_updated_on']) > track_age_thresh:
                stable_tracks.append(t)
                del tracks[tracks.index(t)]
        # update list of tracks that will be processed next frame
        tracks = tracks + updated_tracks + new_tracks
    # all frames have been processed, save all tracks that are longer than min len
    returned_tracks = []
    for st in stable_tracks:
        if len(st['bboxs']) > min_track_len:
            returned_tracks.append(st)
    for t in tracks:
        if len(t['bboxs']) > min_track_len:
            returned_tracks.append(t)
    return returned_tracks

if __name__ == '__main__':

    bb_data = pd.read_csv(input_file, header=None, names=names)
    # pre-process bounding box data
    bb_data['frame'] = bb_data['frame'] + 1     # shift frames to align with ground truth data
    bb_data['x_center'] = bb_data['left'] + (bb_data['width'] / 2)      # compute x of bb centroid
    bb_data['y_center'] = bb_data['top'] + (bb_data['height'] / 2)      # compute y of bb centroid
    # add histograms to 6 rows

    if use_color_histogram:
        bb_data['hist'] = bb_data.apply(lambda row: hist.computeHistogram(row[0], row), axis=1)
    '''
    # TODO make check in order to load correct data if exists, otherwise compute it
    bb_data = pd.read_csv('bb_with_hist.csv')

    # res = hist.computeHistogram(1, getDetectionsByFrame(bb_data, 1)[0])
    # res2 = hist.computeHistogram(3, getDetectionsByFrame(bb_data, 3)[2])


    # print(hist.compareHists(res, res2))
    '''
    result = track(bb_data)

    avgTrackLen = 0
    for r in result:
        avgTrackLen += len(r['bboxs'])
    avgTrackLen /= len(result)
    print('Num Track: ' + str(len(result)))
    print('Avg Track Len: ' + str(avgTrackLen))

    parsetTracks = plotUtil.parseTracks(result)
    plotUtil.showTracksOnImage(parsetTracks)
