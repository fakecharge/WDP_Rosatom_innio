from ultralytics import YOLO 
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
from itertools import combinations, chain

file_path = './bd/ДеталиПоПлануДляРазрешенныхЗаказов.xlsx'
modet_path = "./models/best.pt"


def detect_and_recognition(img):
    # Загрузка моделей YOLO и paddleocr
    model = YOLO(modet_path)
    ocr = PaddleOCR(lang='en')

    # Детектирование областей с текстом с помощью YOLO
    result_yolo = model.predict(img)

    # Лист распознанного текста
    text = []

    # Распознавание текста в каждой области с помощью paddleocr
    for i in range(len(result_yolo)):

        fig, ax = plt.subplots()
        boxes_yolo = result_yolo[i].boxes
        for j in range(len(boxes_yolo)):
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = boxes_yolo.xyxy[j]
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = int(x1_yolo), int(y1_yolo), int(x2_yolo), int(y2_yolo)
            cropped_image = img[y1_yolo:y2_yolo, x1_yolo:x2_yolo]

            result_ocr = ocr.ocr(cropped_image, cls=False)

            if result_ocr[0] != None:
                for idx in range(len(result_ocr)):
                    res = result_ocr[idx]
                    for line in res:
                        print(line)

                # draw result
                result_ocr = result_ocr[0]
                txts = [line[1][0] for line in result_ocr]
                text.append(txts)

                boxes_ocr = [line[0] for line in result_ocr]
                scores = [line[1][1] for line in result_ocr]

                boxes_ocr = np.array(boxes_ocr)
                for i in range(len(boxes_ocr)):
                    # boxes_ocr[i] = np.array(boxes_ocr[i])
                    print(boxes_ocr[i])
                    boxes_ocr[i][:, 1] += y1_yolo
                    boxes_ocr[i][:, 0] += x1_yolo
                    polygon = plt.Polygon(boxes_ocr[i], edgecolor='r', facecolor='none')
                    ax.add_patch(polygon)
                    ax.set_title(txts)
                    ax.text(np.mean(boxes_ocr[i][:, 0]), np.mean(boxes_ocr[i][:, 1]), f'{txts[i]}', color='black',
                            fontsize=16, ha='center', va='center')
                    ax.imshow(img)


            else:
                print("No text detected.")

    ax.set_axis_off()# Убираем оси
    # Обновляем картинку перед сохранением
    plt.draw()
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img = Image.open(buf).copy()

    buf.close()  # Закрываем буфер
    plt.close(fig)  # Закрываем фигуру после сохранения

    return text, img



def similarity(a, b):
    """Возвращает коэффициент схожести между двумя строками."""
    return SequenceMatcher(lambda x: x in " -_.\\/[]'\"", a, b, autojunk=True).ratio()

def find_best_match(df, search_string):
    max_similarity = 0
    best_row_index = -1

    for index, row in df.iterrows():
        combined_text = str(row[0]) + str(row[1])

        search_strings = ' '.join(search_string[0]) # Объединим все найденные строки в одну
        #print(search_strings)
        # Считаем схожесть
        avg_similarity = similarity(combined_text, search_strings)

        if avg_similarity > max_similarity:
            max_similarity = avg_similarity
            best_row_index = index

    return best_row_index

def find_best_match_all_combinations(table, snippets, table_vectors, vectorizer ):
    if table.empty:
        return -1

    # # Объединяем первый и второй столбцы для каждой строки
    # combined_text = table.iloc[:, :2].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    #
    # # Используем TF-IDF для векторизации текста
    # vectorizer = TfidfVectorizer()
    # table_vectors = vectorizer.fit_transform(combined_text)

    best_match_index = -1
    max_similarity = 0

    # Генерируем все возможные склейки отрывков
    for i in range(1, len(snippets) + 1):
        for snippet_combination in combinations(snippets, i):
            # Склеиваем отрывки в одну строку
            combined_snippet = " ".join(chain.from_iterable(snippet_combination))  # Склейка элементов списков
            snippet_vector = vectorizer.transform([combined_snippet])

            # Вычисляем косинусное сходство для каждой строки в таблице
            similarities = cosine_similarity(snippet_vector, table_vectors)

            # Находим строку с максимальным сходством
            current_best_index = np.argmax(similarities)
            current_best_similarity = similarities[0, current_best_index]

            # Обновляем наилучшее совпадение, если текущее сходство больше
            if current_best_similarity > max_similarity:
                best_match_index = current_best_index
                max_similarity = current_best_similarity

    return best_match_index


def find_in_BD(text, df, table_vectors, vectorizer):
    #best_match_index = find_best_match(df, text)
    best_match_index = find_best_match_all_combinations(df, text, table_vectors, vectorizer)

    best_match_row = df.iloc[best_match_index]  # Получаем всю строку по индексу
    print("Строка с максимальным совпадением:")
    print(best_match_row)

    return best_match_row

# проверка работоспособности без web интерфейса
if __name__ == "__main__":
    img = Image.open('d:/mark/train_dataset_rosatom_train/train Росатом/train/imgs/16.JPG')
    plt.imshow(img)
    plt.axis('off')  # Отключаем оси
    plt.show()

    img = img.convert('L')
    img = img.resize((480, 640))
    img = np.array(img)
    text, fig = detect_and_recognition(img)

    print(text)

    best_match = find_in_BD(text, df)


