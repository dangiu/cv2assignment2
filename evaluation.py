import motmetrics as mm
import numpy as np
import pandas as pd

track_data_file = 'tracking.txt'
gt_data_file = 'Video/gt/gt.txt'

track_data_names = ['frame', 'id', 'x_center', 'y_center']
gt_data_names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'confidence', 'x', 'y', 'z']

def getIdsPerFrame(frame_index, data):
    """
    Extracts ids for traks in a specific frame, sorted by id
    :param frame_index:
    :param data:
    :return: list of ids of tracks in specified frame
    """
    rows = data.loc[data['frame'] == frame_index]
    rows = rows.sort_values(by=['id'])
    return rows.id.unique()

def getRowsByFrame(frame_index, data):
    rows = data.loc[data['frame'] == frame_index]
    rows = rows.sort_values(by=['id'])
    return rows

def getPointList(id_list, frame_index, data):
    rows = getRowsByFrame(frame_index, data)
    points = []
    for index, row in rows.iterrows():
        if row['id'] in id_list:
            points.append([row['x_center'], row['y_center']])
    return points

if __name__ == '__main__':
    track_data = pd.read_csv(track_data_file, header=None, names=track_data_names)
    gt_data = pd.read_csv(gt_data_file, header=None, names=gt_data_names)
    # add x_center and y_center to ground truth dataset, they are used to compute metrics
    gt_data['x_center'] = gt_data['bb_left'] + (gt_data['bb_width'] / 2)
    gt_data['y_center'] = gt_data['bb_top'] + (gt_data['bb_height'] / 2)

    #getPointList([1,2,6], 1, track_data)

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in gt_data.frame.unique():
        o_ids = getIdsPerFrame(frame, gt_data)
        h_ids = getIdsPerFrame(frame, track_data)

        # Object related points
        o = np.array(
            getPointList(o_ids, frame, gt_data)
        )

        # Hypothesis related points
        h = np.array(
            getPointList(h_ids, frame, track_data)
        )

        C = mm.distances.norm2squared_matrix(o, h, max_d2=10000)

        acc.update(
            o_ids,  # Ground truth objects in this frame
            h_ids,  # Detector hypotheses in this frame
            C
        )

    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc],
        metrics=mm.metrics.motchallenge_metrics,
        names=['tracking']
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)