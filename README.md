#  YOLOv5 Auto Annotator

Annotate datasets with a semi-trained or fully trained YOLOv5 model

## Prerequisites

```json
Ubuntu >=20.04
Python >=3.7
```

## System dependencies

```sh
sudo apt install python3-dev python3-pip
```

## Python dependencies

```json
cycler==0.11.0
fonttools==4.29.0
kiwisolver==1.3.2
lxml==4.6.4
numpy==1.21.4
opencv-contrib-python==4.5.5.62
opencv-python==4.5.5.62
packaging==21.3
Pillow==9.0.0
pyparsing==3.0.7
python-dateutil==2.8.2
six==1.16.0
tqdm==4.62.3
```

Install with the following command - 

```sh
pip3 install -r requirements.txt
```

## Run the application

Execute `annotate.py` in the following format - 

```sh
usage: annotate.py [-h] [--viewmode] [--imgdir IMGDIR] [--annodir ANNODIR] [--confThreshold CONFTHRESHOLD] [--nmsThreshold NMSTHRESHOLD] [--width WIDTH] [--height HEIGHT] [--onnx_path ONNX_PATH] [--labels_path LABELS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --viewmode            Toggle View Mode
  --imgdir IMGDIR       Directory of images
  --annodir ANNODIR     Directory of annotations
  --confThreshold CONFTHRESHOLD
                        Class confidence
  --nmsThreshold NMSTHRESHOLD
                        NMS threshold
  --width WIDTH         Width of network input
  --height HEIGHT       Height of network input
  --onnx_path ONNX_PATH
                        Path to onnx file
  --labels_path LABELS_PATH
                        Path to labels file
```

Example - 

```sh
python3 annotate.py --imgdir /home/kn1ght/Documents/images --annodir annotations --onnx_path models/YOLOv5s/yolov5s.onnx --labels_path models/YOLOv5s/coco.names --viewmode
```
