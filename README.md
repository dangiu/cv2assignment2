# Object Detection and Tracking in Practice
I suggest reading this file from the Github repository [cv2assignment2](https://github.com/daniele122008/cv2assignment2) where is formatted and displayed correctly.
### Demo
A demo of the implementation can be found [here](https://www.insert.real.link)
### Repository Organization
The repository is organized as follow:
```
root/
|-- Video
|   |-- gt
|   |   |-- gt.txt
|   |
|   |-- img1
|   |   |-- 000001.jpg
|   |   |-- 000002.jpg
|   |   |-- 000003.jpg
|   |   |-- ...
|
|-- bb_out.csv
|-- detection.py
|-- detection.txt
|-- evaluation.py
|-- histograms.py
|-- iou-tracking.py
|-- README.md
|-- tracking.py
|-- tracking.txt
|-- tracking_no_hist.txt
|-- utility.py
```

### Implementation
All the implementation is written in **Python 3** which must be installed on the system in order to be able to execute the code.

#### Required Libraries
* **Numpy** for numerical calculation
* **Pandas** for CSV file mangement
* **GluonCV** provides pretrained Faster-CNN
* **OpenCV** to visualize result of the algorithm
* **MotMetrics** to compute the metrics used in the Multi Object Tracking challenge

#### Performing Detection
In order to perform detection is sufficient to execute the script `detection.py` making sure that the attribute `save_to_file` is set to `True`. Be aware that the process might take a while (as a suggestion, just let it run over night on a laptop and by the morning the detection should be completed) as the current version of this script does not take advantage of discrete GPUs in order to run faster.

The output of this procedure is the file `bb_out.csv`, which is required in order to perform the tracking part. In this repository we include the file obtained, for simplicity.

The CSV file has the following columns:
1. Frame to which the detection belongs (the first frame has number `0`).
2. ID of the detection for the current frame, this is just a counter ranging from `0` to `N`, where `N` is the number of detections in the current frame minus 1.
3. Left most point of the bounding box (X axes).
4. Top most point of the bounding box (Y axes).
5. Bounding box width.
6. Bounding box height.


##### Generate detection.txt
In order to generate the output file `detection.txt` required for the delivery, execute the script `utility.py`. The ouput file will be generated in the root folder. The script `detection.py` does not generate the actual output file, due to the fact that initially it was done this way and reperforming all the detections just to change the output format of the file is too time consuming, given the unavailability of powerful hardware.

#### Perform Tracking
In order to perform different types of tracking and visualize the result on screen, multiple options are available:
1. To perform IOU-based tracking, execute the script `iou-tracking.py`.
2. To perform centroid-based tracking, execute the script `tracking.py`. 
3. To perform centroid-based tracking with the reidentification model based on color histograms, execute the script `tracking.py` while making sure that the attribute `use_color_histogram` is set to `True`. While running in this mode, around 30-40 seconds can elapse before any output is visualized, because the color histograms for all the bounding boxes must be computed before initializing the tracking process.

#### Generate tracking.txt
Using options 2 and 3 of the tracking methods, is possible to generate the output file `tracking.txt` required for the delivery by ensuring that the attribute `save_to_file` is set to `True`. The output file will be generated in the root folder.


#### Evaluating Performance
In order to evaluate the performance of tracking  methods 2 and 3, the script `evaluation.py` must be executed, making sure that the attributes `track_data_file` and `gt_data_file` are set respectively to the file containing the tracking data in the delivery format, and the ground truth data.
After the execution the MOT metrics will be printed out.

#### Additional Information
* The file `tracking_no_hist.txt` contains the tracking data for method 2 in order to simplify the generation of the scores without needing to reperform the tracking.
* The script `histograms.py` is a library containing functions to compute the color histogram given a bounding box and also to compare two color histograms.
* The naming schema of directories and file is important for the correct execution of the code.
