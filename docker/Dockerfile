FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ADD ./docker/apt.packages /tmp/apt.packages

RUN apt-get update && \
    apt-get install -y $(awk '{print $1}' /tmp/apt.packages) && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update

RUN apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev

RUN wget https://bootstrap.pypa.io/get-pip.py &&  \
	 python3.7 get-pip.py && \
	 ln -s /usr/bin/python3.7 /usr/local/bin/python3

ADD ./requirements.txt /tmp/requirements.txt

RUN pip install $(grep 'torch==' /tmp/requirements.txt) && \
    pip install $(grep 'numpy==' /tmp/requirements.txt)
RUN pip install -r /tmp/requirements.txt

WORKDIR /app

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait" 
