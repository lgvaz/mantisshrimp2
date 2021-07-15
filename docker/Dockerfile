FROM python:3.8

RUN pip install PyYAML>=5.1 -U
RUN pip install datascience -U

RUN pip install torchtext==0.9.0 -U
RUN pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -U

RUN pip install fastai==2.3.1 -U

RUN pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] -U
RUN pip install git+git://github.com/airctic/icedata.git -U
RUN pip install yolov5-icevision -U 
RUN pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html -U
RUN pip install mmdet==2.13.0 -U
RUN pip install ipywidgets -U