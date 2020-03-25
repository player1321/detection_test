ARG PYTORCH="0.4.1"
ARG CUDA="9"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# FROM pytorch/pytorch:latest

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# new add
ENV DEBIAN_FRONTEND noninteractive  
RUN apt-get clean
RUN sed -i "s|http://archive.ubuntu.com|http://mirrors.163.com|g" /etc/apt/sources.list && rm -Rf /var/lib/apt/lists/* && apt-get -y update 
###

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install cocoapi
ENV COCOAPI=/workspace/cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
RUN cd $COCOAPI/PythonAPI
RUN make
RUN python setup.py install --user

# Install centernet
ENV CenterNet_ROOT=/workspace
RUN git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
RUN cd $CenterNet_ROOT/CenterNet
RUN pip install -r requirements.txt
RUN cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
RUN ./make.sh
RUN cd $CenterNet_ROOT/src/lib/external
RUN make

# load my code
ENV My_ROOT=/workspace
RUN git clone https://github.com/player1321/detection_test.git $My_ROOT
RUN cp $My_ROOT/coco.py $CenterNet_ROOT/CenterNet/src/lib/datasets/dataset/coco.py
RUN cp $My_ROOT/debugger.py $CenterNet_ROOT/CenterNet/src/lib/utils/debugger.py
RUN cp $My_ROOT/ctdet.py $CenterNet_ROOT/src/lib/detectors/ctdet.py

# data preprocess
RUN cd $My_ROOT
RUN python draft.py /home/data

# Install mmdetection
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
# WORKDIR /mmdetection
# ENV FORCE_CUDA="1"
# RUN pip install --no-cache-dir -e .
