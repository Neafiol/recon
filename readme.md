Установка
==========
1. `python3.6 -m venv venv` - добавим виртуальную среду
2. `./venv/bin/pip install cython` - установим cython
3. `./venv/bin/pip install -e tf-pose-estimation/` - установим tf-pose-estimation в качестве пакета, с кодом в исходной дирректори
4. `./venv/bin/pip install -r requerements.txt` - установим остальные зависимости

Разметка
=========
1. Поместите дамп бд в директорию проекта (photos.txt)
2. Для загрузки фото и подготовки файла с векторизованными данными вызовете `./venv/bin/python photos_preparing.py`,
после чего в папку data буду выгружены все фото и в файл photos.pk будут сохранены их векторизованные представления
3. Для дообучения модели вызовите `./venv/bin/python learning.py`
4. Для зазметки файла photos.txt (определение типа элемента для всех строк где он не указан)
выполните `./venv/bin/python predicting.py`, после чего будет сгенерирован файл fphotos.csv который можно импортить в вашу БД