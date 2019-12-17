import pandas as pd
import numpy as np
import gluoncv
import math
import plotUtil

# parameters
input_file = 'bb_out.csv'
names = ['frame', 'bbox', 'left', 'top', 'width', 'height']
min_score_thresh = 0.005
track_age_thresh = 16
min_track_len = 4

output_name = 'tracking.csv'

def getDetectionsByFrame(detections, frame_index):
    '''

    :param detections: dataframe containing detections data
    :param frame_index: frame for which the detection are wanted
    :return: list of detection, each in the form:
            frame, left, top, width, height, x_center, y_center
    '''
    df = detections.loc[detections['frame'] == frame_index]
    return df.drop(['bbox'], 1).values.tolist()

def getFrameList(detections):
    return detections.frame.unique()

def computeDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def computeMatch(track, detection):
    last_track_detect = track['bboxs'][-1]
    d = computeDistance(last_track_detect[-2], last_track_detect[-1], detection[-2], detection[-1])
    return 1 / d

def computeBestMatch(tracks, detections):
    best_score = 0
    best_track = None
    best_detection = None
    for t in tracks:
        for d in detections:
            score = computeMatch(t, d)
            #print(score)
            if score > best_score:
                best_score = score
                best_track = t
                best_detection = d
    return best_score, best_track, best_detection

def track(detections):
    tracks = []
    stable_tracks = []
    for frame in getFrameList(detections):
        updated_tracks = []
        frame_detections = getDetectionsByFrame(detections, frame)
        # matching part
        no_match = False
        while len(frame_detections) > 0 and not no_match:
            score, t, d = computeBestMatch(tracks, frame_detections)
            print('Best: ' + str(score))
            if score < min_score_thresh:
                no_match = True
            else:
                # add detection to track (best match pair)
                #tracks[tracks.index(t)]['bboxs'].append(d)
                #tracks[tracks.index(t)]['last_uodated_on'] = frame
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
            # TODO IMPLEMENT
            # otherwise create new track
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
    # PRE PROCESS BOUNDING BOX DATA
    bb_data['frame'] = bb_data['frame'] + 1     # shift frames to align with ground truth data
    bb_data['x_center'] = bb_data['left'] + (bb_data['width'] / 2)      # compute x of bb centroid
    bb_data['y_center'] = bb_data['top'] + (bb_data['height'] / 2)      # compute y of bb centroid

    result = track(bb_data)

    avgTrackLen = 0
    for r in result:
        avgTrackLen += len(r['bboxs'])
    avgTrackLen /= len(result)
    print('pause')

    parsetTracks = plotUtil.parseTracks(result)
    plotUtil.showTracksOnImage(parsetTracks, None)

    #TODO make utility to show what is beign traked on the image!
    # each track with differetn color bonding box, so you can see the swith etc etc

