FROM pytorch/pytorch


RUN apt-get update && apt-get install -y \
    gcc \
    libmysqlclient-dev \
	libgl1-mesa-glx \
	libglib2.0-0 \
	mysql-client
	
RUN pip install --upgrade pip

COPY environment.yaml /workspace

RUN conda env create --file environment.yaml

RUN echo "source activate lesion" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init bash && \
    echo "conda activate lesion" >> ~/.bashrc && \
    pip install django==3.2.22 django-cors-headers==4.1.0 \
    django-extensions==3.2.3 \
    django-silk==5.0.3 \
    djangorestframework==3.14.0 \
    mysqlclient==2.1.1 \
	setproctitle \
	natsort \
	scikit-image==0.19.2 \
	scikit-learn==1.0.2 \
	efficientnet-pytorch==0.7.1 \
	silence-tensorflow==1.2.1 \
	tensorboard==2.2.2 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.1 \
    tensorflow==2.2.0 \
    tensorflow-estimator==2.2.0 \
    tensorflow-io-gcs-filesystem==0.33.0 \
	openpyxl==3.1.2 \
	opencv-python==4.7.0.68 \
	opencv-python-headless==4.6.0.66 \
	#torch==1.10.1 \
	#torchinfo==1.7.2 \
	#torchvision==0.11.2 \
	tqdm==4.64.0

RUN pip install ultralytics==8.0.25
#RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
COPY efficientnet-b0-355c32eb.pth /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth

COPY . /workspace

WORKDIR /workspace/server
ENTRYPOINT ["conda", "run", "-n", "lesion", "python", "manage.py", "runserver", "0.0.0.0:8000"]
