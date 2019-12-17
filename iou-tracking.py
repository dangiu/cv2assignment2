import pandas as pd
import numpy as np
import gluoncv

#Parameters
input_file = 'bb_out.csv'
names = ['frame', 'bbox', 'left', 'top', 'width', 'height']
gt_data_path = 'Video/gt/gt.txt'
gt_names = ['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z']

output_name = 'iou_tracking.csv'

def iou(bboxA, bboxB):
    #convert bboxA to ndarry
    Aminx = bboxA[2]
    Aminy = bboxA[3]
    Amaxx = Aminx + bboxA[4]
    Amaxy = Aminy + bboxA[5]
    # convert bboxB to ndarry
    Bminx = bboxB[2]
    Bminy = bboxB[3]
    Bmaxx = Bminx + bboxB[4]
    Bmaxy = Bminy + bboxB[5]
    # create arrays
    bbA = np.asarray([Aminx, Aminy, Amaxx, Amaxy])
    bbA = bbA.reshape((1,4))
    bbB = np.asarray([Bminx, Bminy, Bmaxx, Bmaxy])
    bbB = bbB.reshape((1,4))
    # compute IOU using gluoncv
    result = gluoncv.utils.bbox.bbox_iou(bbA, bbB)
    return result[0][0]

def getDetectionsByFrame(detections, frameIndex):
    df = detections.loc[detections['frame'] == frameIndex]
    return df.values.tolist()

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
                best_match = max(frameDetect, key=lambda x: iou(ta['bboxes'][-1], x))
                if iou(ta['bboxes'][-1], best_match) >= minIOU:
                    #if threshold is surpassed update track and remove from available detection (because it was assigned to a track)
                    ta['bboxes'].append(best_match)
                    tracksUpdated.append(ta)
                    del frameDetect[frameDetect.index(best_match)]

            if len(tracksUpdated) == 0 or ta is not tracksUpdated[-1]:
                # if track was not updated finish track when the conditions are met
                if len(ta['bboxes']) >= minLen:
                    tracksFinished.append(ta)

        # create new tracks
        tracksNew = [{'bboxes': [det], 'start_frame': frame} for det in frameDetect]
        tracksActive = tracksUpdated + tracksNew

    # finish all remaining active tracks that meet conditions
    tracksFinished += [ta for ta in tracksActive if len(ta['bboxes']) >= minLen]



    return tracksFinished


if __name__ == '__main__':
    bbData = pd.read_csv(input_file, header=None, names=names)
    bbData['frame'] = bbData['frame'] + 1 #shift frames to align with ground truth data
    gtData = pd.read_csv(gt_data_path, header=None, names=gt_names)

    finished_tracks = track(bbData)
    avgTrackLen = 0;
    for t in finished_tracks:
        avgTrackLen += len(t['bboxes'])
    avgTrackLen /= len(finished_tracks)
    print('IOU Approach')
    print('N of extracted tracks: ' + str(len(finished_tracks)))
    print('Avg track len: ' + str(avgTrackLen))

    print('Ground Truth Data')
    print('N of tracks: ' + str(len(gtData.id.unique())))

    gtAvgTrackLen = 0;
    for i in gtData.id.unique():
        iData = gtData.loc[gtData['id'] == i]
        gtAvgTrackLen += len(iData.frame.unique())
    gtAvgTrackLen /= len(gtData.id.unique())

    print('Avg track len: ' + str(gtAvgTrackLen))

