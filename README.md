# vehicle-counting-yolo-sort-python

This project imlements the following tasks in the project: 

1. Vehicle counting
2. Lane segmentation
3. Lane change detection
4. speed estimation
5. Dumps all these details into a CSV file

![SCREENSHOT](https://github.com/bamwani/car-counting-yolo-sort-python/blob/master/Screenshot1.png)


This project use YOLOv3 for Vehicle detection and SORT(Simple Online and Realtime Tracker) for vehicle tracking

# To run the project:

1. Download the code or simply run: ``` git clone https://github.com/bamwani/car-counting-yolo-sort-python ``` in the terminal

2. To install required dependencies run:
```
$ pip install -r requirements.txt
```

3. Make sure you change the line of detection and lane segmentation according to your video and fine tune the threshold and confidence for YOLO model

4. Run ```main.py -input /path/to/video/file.avi -output /path/for/output/video/file.avi -yolo /path/to/YOLO/directory/``` 



I will keep making commits to improve the speed detection of vehicles.
Any pull request for improvement is highly welcomed.



# References:


[YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

[SORT Algorithm](https://github.com/abewley/sort)

[Reference for combining YOLO and SORT](https://github.com/guillelopez/python-traffic-counter-with-yolo-and-sort)
