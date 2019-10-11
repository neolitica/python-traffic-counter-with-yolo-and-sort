# People and face tracker.

Developed by [@ironcadiz](https://github.com/ironcadiz) for [neolitica.ai](neolitica.ai)

This is a live people and face tracker built with pytorch and opencv. It can process live video from a camera or a full video given as input. Object detection is based on yolov3 and the face, age and gender classification are based on open source models trained on the aidience dataset which were ported from caffe to pytorch with [MMdnn](https://github.com/microsoft/MMdnn).

## Setup and requirements

This has only been tested on Python 3.6.8 with CUDA 10.0

Install the requirements
```bash
pip install -r requirements.txt
```
Download the yolov3 weights [from here](https://pjreddie.com/media/files/yolov3.weights) and place them in the `yolo-coco` folder.

# Usage

To see the full arguments use.

```bash
python main.py --h
```

To run from webcam leave the `--input` argument blank

Example run on webcam:

```bash
python main.py --output output/test.mp4 -do results.csv  --yolo yolo-coco   -ct -ch -sh
```

# References

* Pytorch yolo implementation by [ayooshkathuria](https://github.com/ayooshkathuria/pytorch-yolo-v3)
* Object detection based on work by [guillelopez](https://github.com/guillelopez/python-traffic-counter-with-yolo-and-sort)