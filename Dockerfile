FROM nvcr.io/nvidia/pytorch:22.07-py3
ENV NVIDIA_VISIBLE_DEVICES all
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
EXPOSE 8888

RUN apt update
RUN apt upgrade -y
RUN apt install graphviz -y

COPY requirements.txt . 
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY kaggle.json .
RUN mkdir /root/.kaggle \
    && cp kaggle.json -d /root/.kaggle/kaggle.json \
    && chmod 600 /root/.kaggle/kaggle.json