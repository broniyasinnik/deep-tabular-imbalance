FROM catalystteam/catalyst:v21.11-pytorch-1.8.1-cuda11.1-cudnn8-devel
WORKDIR /bigdata
COPY requirements.txt requirements.txt
COPY src src
VOLUME ["/bigdata/experiments/"]
EXPOSE 5000
RUN python -m pip install -r requirements.txt