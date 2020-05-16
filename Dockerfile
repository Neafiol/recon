FROM tensorflow/tensorflow:version-1.14.0


WORKDIR /home/petr/Documents/Projects/Recognition

COPY requirements.txt requirements.txt
RUN pip3 --upgrade pip
RUN pip3 install cython
RUN pip3 install -r requirements.txt
RUN mkdir data

COPY photos_preparing.py learning.py predicting.py ./

ENTRYPOINT ["python3","predicting.py"]