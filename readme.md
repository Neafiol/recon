Установка
==========
* `sudo apt-get install build-essential libssl-dev libffi-dev python-dev lib32ncurses5-dev python-snappy libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime` - скачаем зависимости
* `export LLVM_CONFIG=/usr/bin/llvm-config-7`
* `git https://github.com/Neafiol/recon` скачаем код проекта
* `cd recon`
* `pip3 install -r requirements.txt` установим зависимости
* `git clone https://www.github.com/ildoonet/tf-pose-estimation` - скачаем репозиторий для нахождения ключевых точек на фотографиях
* `cd tf-pose-estimation`
* `pip3 install cython` - скачаем cython, необходимый для работы tensorflow
* `sudo python setup.py install` установим скачанную библиотеку
*  `cd models/graph/cmu`
* `bash download.sh` - скачаем модельки, которые не влезли на GB

Разметка
=========
1. Поместите дамп бд в директорию проекта (photos.txt)
2. Для загрузки фото и подготовки файла с векторизованными данными вызовете `python3 photos_preparing.py`,
после чего в папку data буду выгружены все фото и в файл photos.pk будут сохранены их векторизованные представления
3. Для дообучения модели вызовите `python3 learning.py`
4. Для разметки файла photos.txt (определение типа элемента для всех строк где он не указан)
выполните `python3 predicting.py`, после чего будет сгенерирован файл fphotos.csv который можно импортить в вашу БД. Если фотография не распозналась, поле тип элесента останеться пустым.