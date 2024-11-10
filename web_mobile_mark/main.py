import streamlit as st
from PIL import Image
import text_detect_recog
import numpy as np
import pandas as pd
import connect_1c
from sklearn.feature_extraction.text import TfidfVectorizer


data_loaded = False

# Компонент для получения изображения с камеры
image_file = st.camera_input("Сделайте снимок")

# Загрузка базы из 1C или локально из файла
local_file_path = './bd/ДеталиПоПлануДляРазрешенныхЗаказов.xlsx'
@st.cache_data
def load_data():
    global data_loaded
    global suggestions  # Делаем suggestions глобальной переменной
    if not data_loaded:
        try:
            json_data = connect_1c.getOdataRest1C('ДеталиПоПлануДляРазрешенныхЗаказов')
            df = pd.json_normalize(json_data)
        except:
            # Загружаем данные из файла
            df = pd.read_excel(local_file_path) #, usecols=[0]
        print('vectorized----------------------------------][][]')
        # Объединяем первый и второй столбцы для каждой строки
        combined_text = df.iloc[:, :2].apply(lambda row: ' '.join(row.astype(str)), axis=1)

        # Используем TF-IDF для векторизации текста
        vectorizer = TfidfVectorizer()
        table_vectors = vectorizer.fit_transform(combined_text)
    data_loaded = True

    return df, table_vectors, vectorizer

if image_file is not None:
    # Если изображение было получено, отображаем его
    image = Image.open(image_file)


    #st.image(image, caption="Сделанное изображение")
    image = np.array(image)
    text, img_recogn = text_detect_recog.detect_and_recognition(image)
    if not text:
        st.header('Маркировка не найдена')
    else:
        # Найденная строка
        st.header(text[0])

        # Загружаем базу для поиска и для автодополнения при ручном вводе
        df, table_vectors, vectorizer = load_data()

        # Ищем в базе билжайший артикть и серийный номер
        best_match = text_detect_recog.find_in_BD(text, df, table_vectors, vectorizer)



        # Получаем список слов из первого столбца, оставляя только уникальные строки
        suggestions = list(set(df[df.columns[0]].tolist()))
        suggestions = [s[1:-1] for s in suggestions]

        article = best_match.iloc[0]
        article = article[1:-1]
        # ДетальАртикул с автодополнением
        selected_options = st.selectbox("ДетальАртикул", suggestions, index=suggestions.index(article) if article in suggestions else 0)

        serial = best_match.iloc[1]
        st.text_input("ПорядковыйНомер", value=int(serial))

        st.write("ДетальНаименование: " + best_match.iloc[2][1:-1])
        st.write("ЗаказНомер: " + best_match.iloc[3][1:-1])
        st.write("СтанцияБлок: " + best_match.iloc[4][1:-1])

        if st.button("Отправить"):
            data = {
                 'ДетальАртикул': '"' + article + '"',
                 'ПорядковыйНомер': serial,
                 'ДетальНаименование': best_match.iloc[2],
                 'ЗаказНомер': best_match.iloc[3],
                 'СтанцияБлок': best_match.iloc[4],
            }
            ans = connect_1c.setOdataRest1C('ДеталиПоПлануДляРазрешенныхЗаказов', data)
            if not ans:
                st.warning('Сервер не отвечает')


        st.image(img_recogn, caption="Размеченное изображение")
