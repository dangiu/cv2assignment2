import pandas as pd
import numpy as np
import gluoncv
import utility

#Parameters
input_file = 'bb_out.csv'
names = ['frame', 'bbox', 'left', 'top', 'width', 'height']
gt_data_path = 'Video/gt/gt.txt'
gt_names = ['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z']

output_name = 'iou_tracking.csv'

def iou(bboxA, bboxB):
    #frame, id, left, top, width, height
    #convert bboxA to ndarry
    Aminx = bboxA['left']
    Aminy = bboxA['top']
    Amaxx = Aminx + bboxA['width']
    Amaxy = Aminy + bboxA['height']
    # convert bboxB to ndarry
    Bminx = bboxB['left']
    Bminy = bboxB['top']
    Bmaxx = Bminx + bboxB['width']
    Bmaxy = Bminy + bboxB['height']
    # create arrays
    bbA = np.asarray([Aminx, Aminy, Amaxx, Amaxy])
    bbA = bbA.reshape((1,4))
    bbB = np.asarray([Bminx, Bminy, Bmaxx, Bmaxy])
    bbB = bbB.reshape((1,4))
    # compute IOU using gluoncv
    result = gluoncv.utils.bbox.bbox_iou(bbA, bbB)
    return result[0][0]

def getDetectionsByFrame(detections, frame_index):
    # frame, id, left, top, width, height, x_center, y_center, hist
    df = detections.loc[detections['frame'] == frame_index]
    return df.drop(['bbox'], 1).to_dict('records')

def track(detections, minIOU=0.5,minLen=2):
    tracksActive = []
    tracksFinished = []

    frameList = detections.frame.unique()
    for frame in frameList:
        frameDetect = getDetectionsByFrame(detections, frame)
        tracksUpdated = []
        for ta in tracksActive:
            # get detection with highest IOU
            if len(frameDetect) > 0:
                best_match = max(frameDetect, key=lambda x: iou(ta['bboxs'][-1], x))
                if iou(ta['bboxs'][-1], best_match) >= minIOU:
                    #if threshold is surpassed update track and remove from available detection (because it was assigned to a track)
                    ta['bboxs'].append(best_match)
                    tracksUpdated.append(ta)
                    del frameDetect[frameDetect.index(best_match)]

            if len(tracksUpdated) == 0 or ta is not tracksUpdated[-1]:
                # if track was not updated finish track when the conditions are met
                if len(ta['bboxs']) >= minLen:
                    tracksFinished.append(ta)

        # create new tracks
        tracksNew = [{'bboxs': [det], 'start_frame': frame} for det in frameDetect]
        tracksActive = tracksUpdated + tracksNew

    # finish all remaining active tracks that meet conditions
    tracksFinished += [ta for ta in tracksActive if len(ta['bboxs']) >= minLen]

    return tracksFinished


if __name__ == '__main__':
    bbData = pd.read_csv(input_file, header=None, names=names)
    bbData['frame'] = bbData['frame'] + 1 #shift frames to align with ground truth data

    finished_tracks = track(bbData)
    avgTrackLen = 0
    for t in finished_tracks:
        avgTrackLen += len(t['bboxs'])
    avgTrackLen /= len(finished_tracks)
    print('IOU Approach')
    print('N of extracted tracks: ' + str(len(finished_tracks)))
    print('Avg track len: ' + str(avgTrackLen))

    ft = utility.parseTracks(finished_tracks)
    utility.showTracksOnImage(ft)

