# Web решения

Построенно с использованием streamlit.<br>
Имеет лаконичный интерфейс для удобства работы.

### Возможности:

- **получение изображения с камеры**
- **обнаружение текстовых полей при помощи Yolo**
- **распознавание текста при помощи paddleocr**
- **поиск по базе похожих строк**
- **вывод описания детали с возможностью ручной коррекции артикля и порядкового номера**
- **кнопка отправки в 1С**

### Запуск

Для запуска используйте команду ```streamlit run main.py```<br>
web интерфейс Будет доступен по адресу: ```localhost:8502```<br>
Можно протестировать развернутую версию по адресcу: https://mark.innino.keenetic.pro/

### Варианты использования:

- **неттоп + web камера**
- **работа через браузер мобильного телефона**
- **apk приложение для android (загруженно в облако, ссылка в лк.)**
- **возможность работы как глобальной так и в локальной сети**

### Структура проекта:

- **fonts** - шрифты в формате *.ttf, которые будут доступны для генерации
- **bd** - должен быть файл ДеталиПоПлануДляРазрешенныхЗаказов.xlsx если сервер 1С не ответит, работа будет по локальной базе
- **modeles** - обученные модели
- **main.py** - web интерфейс
- **text_detect_recog.py** - обнаружение-распознавание-поиск по базе 

### Основные функции:

- **detect_and_recognition** - обнаружение и распознавание текста
- **find_best_match** - поиск по базе наилучшего соответсвия артикль+номер
- **find_best_match_all_combinations** - тоже, для различный вариантов склейки обнаруженных фрагментов текста + векторизация

### Интеграция с 1С
- **connect_1C.py** - Получение базы, отправка наименования детали	
	