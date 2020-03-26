ARG PYTORCH="0.4.1"
ARG CUDA="9"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# # FROM pytorch/pytorch:latest
#######
# FROM 10.9.0.187/algo_team/cuda10.0-cudnn7.4.2-dev-ubuntu16.04-opencv4.1.1-tensorflow1.13-openvino2020r1-workspace

# 创建默认目录，训练过程中生成的模型文件、日志、图必须保存在这些固定目录下，训练完成后这些文件将被保存
RUN mkdir -p /project/train/src_repo && mkdir -p /project/train/result-graphs && mkdir -p /project/train/log && mkdir -p /project/train/models
#######
ENV CODE_ROOT=/project/train/src_repo
ENV RESULT_ROOT=/project/train/result-graphs
ENV LOG_ROOT=/project/train/log
ENV MODEL_ROOT=/project/train/models

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
ENV COCOAPI=$CODE_ROOT/cocoapi
RUN git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
RUN cd $COCOAPI/PythonAPI
# RUN make
RUN python setup.py build_ext --inplace
RUN rm -rf build
RUN python setup.py install --user

# Install centernet
ENV CenterNet_ROOT=$CODE_ROOT
RUN git clone https://github.com/xingyizhou/CenterNet $CenterNet_ROOT
RUN cd $CenterNet_ROOT/CenterNet
RUN pip install -r requirements.txt
RUN cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
RUN ./make.sh
RUN cd $CenterNet_ROOT/src/lib/external
RUN make

# load my code
ENV My_ROOT=$CODE_ROOT
RUN git clone https://github.com/player1321/detection_test.git $My_ROOT
RUN cp $My_ROOT/my_code/coco.py $CenterNet_ROOT/CenterNet/src/lib/datasets/dataset/coco.py
RUN cp $My_ROOT/my_code/debugger.py $CenterNet_ROOT/CenterNet/src/lib/utils/debugger.py
RUN cp $My_ROOT/my_code/ctdet_det.py $CenterNet_ROOT/src/lib/detectors/ctdet.py
RUN cp $My_ROOT/my_code/ctdet_data.py $CenterNet_ROOT/src/lib/datasets/sample/ctdet.py

# # data preprocess
# RUN cd $My_ROOT
# RUN python draft.py /home/data

# Install mmdetection
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
# WORKDIR /mmdetection
# ENV FORCE_CUDA="1"
# RUN pip install --no-cache-dir -e .
