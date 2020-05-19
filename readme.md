Установка
==========
```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install git  libsm6 libxext6 libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime build-essential libssl-dev libffi-dev python-dev lib32ncurses5-dev python-snappy
export LLVM_CONFIG=/usr/bin/llvm-config-7
curl -O https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
sha256sum Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
conda install -c conda-forge tensorflow=1.14

git clone https://github.com/Neafiol/recon
pip install cython
cd recon
mkdir data
git clone https://www.github.com/ildoonet/tf-pose-estimation
pip3 --default-timeout=1000 install -r requerements.txt
cd tf-pose-estimation
python3 setup.py install
sudo pip install -e .
cd models/graph/cmu
bash download.sh
```

Разметка
=========
0. Заполните файл config.conf
    * _photo_url_ - номер колонки с ссылкой на фото
    * _result_ - номер колонки с результатом 
1. Поместите дамп бд в директорию проекта (photos.txt)
2. Для загрузки фото и подготовки файла с векторизованными данными вызовете `python3 photos_preparing.py`,
после чего в папку data буду выгружены все фото и в файл photos.pk будут сохранены их векторизованные представления
3. Для дообучения модели вызовите `python3 learning.py`
4. Для разметки файла photos.txt (определение типа элемента для всех строк где он не указан)
выполните `python3 predicting.py`, после чего будет сгенерирован файл fphotos.csv который можно импортить в вашу БД. Если фотография не распозналась, поле тип элесента останеться пустым.