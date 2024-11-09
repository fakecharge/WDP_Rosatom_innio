from warnings import catch_warnings

from PIL import Image
import matplotlib.font_manager as fm
import streamlit as st
import numpy as np
import random
import os
from tqdm import tqdm
import time
import zipfile

import image_imit

size = [128,128]


fonts_path = './fonts/' # путь к папке со шрифтами
path_letters = './res/letters/' # путь к папке с результирующими изображениями букв и их эталонов
path_letters_etalon = './res/letters_etalon/'
path_words = './res/words/' # путь к папке с результирующими изображениями слов
path_words_etalon = './res/words_etalon/'

def create_zip(folder_path):
    zip_file_path = f"{folder_path}.zip"
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Добавляем файл в архив с относительным путем
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
    return zip_file_path

# Ввод диапазонов параметров
st.sidebar.header("Настройки генерации")

# Создаем строку с прописными и строчными буквами, а также цифрами
#default_text = ''.join(chr(i) for i in range(65, 91)) + ''.join(chr(i) for i in range(97, 123)) + ''.join(str(i) for i in range(10))
default_text = '0123456789AIKLMOPZ[]aimnxАБВГЕЗИКМНОПРСТУФЭабвиклмнорстч'
symbols = st.sidebar.text_area("Символы", value=default_text)

generate_by_letter = st.sidebar.checkbox("Генерировать одну букву", value=True)

# fonts = [f.name for f in fm.fontManager.ttflist] # Получаем список всех шрифтов в системе
fonts = []
for filename in os.listdir(fonts_path):
    if filename.lower().endswith(('.ttf')): # Проверяем расширения
        fonts.append(filename[0:-4])

# Создаем выпадающий список для выбора шрифта
default_font = "GOST"
selected_font = st.sidebar.selectbox("Шрифт:", fonts, index=fonts.index(default_font) if default_font in fonts else 0)
selected_font = fonts_path + selected_font + '.ttf'

scale_factor_range = st.sidebar.slider("Глубина символа", 0.1, 3.0, (1.0, 2.0))
symbpl_angle_range = st.sidebar.slider("Угол поворота символа", 0, 180, (0, 45))
foto_angle_range = st.sidebar.slider("Отклонение угла съемки от нормали", 0, 90, (0, 45))
light_angle_range = st.sidebar.slider("Отклонение угла освещения от нормали", 0, 90, (0, 70))
light_power_range = st.sidebar.slider("Яркость освещения", 0.5, 1.5, (0.9, 1.1))

# Кнопка для создания архива
if st.sidebar.button("Упаковать сгенерированные изображения"):
    folder_to_zip = './res'  # Папка, которую нужно упаковать
    zip_file = create_zip(folder_to_zip)

    # Предоставляем возможность скачать архив
    with open(zip_file, "rb") as f:
        st.sidebar.download_button(
            label="Скачать архив",
            data=f,
            file_name=os.path.basename(zip_file),
            mime="application/zip"
        )

    st.success("Архив создан и готов к скачиванию!")

st.header("Примеры сгенерированных изображений")
cols = st.columns(3)  # будет 3 столбца с картинками

# Генерация тестовых изображений
num_examples = 1
if generate_by_letter:
    num_examples = 6

i=0
st.write("")  # Добавляем пустую строку
for _ in range(num_examples):
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    symbpl_angle = random.choice([-1, 1]) * np.random.uniform(symbpl_angle_range[0], symbpl_angle_range[1])
    #light_azdeg = np.random.uniform(light_angle_range[0], light_angle_range[1])
    light_altdeg = np.random.uniform(light_angle_range[0], light_angle_range[1])
    elev = random.choice([-1, 1]) * np.random.uniform(foto_angle_range[0], foto_angle_range[1])
    azim = random.choice([-1, 1]) * np.random.uniform(foto_angle_range[0], foto_angle_range[1])
    light_power = np.random.uniform(light_power_range[0], light_power_range[1])

    light_azdeg = np.random.random()*360
    if generate_by_letter:
        leter = random.choice(symbols)
    else:
        leter = image_imit.get_random_cell_from_first_column()

    img = image_imit.generate(leter, selected_font, scale_factor, symbpl_angle, light_azdeg, light_altdeg, elev, azim, light_power)

    cols[i % 3].image(img, use_container_width=True) #use_column_width
    i = i+1




num_generations = st.number_input('Количество генераций:', min_value=1, max_value=1000, value=10)

if st.button('Запустить генерацию'):
    st.info('Генерация запуущенна...')
    if not os.path.exists(path_letters):
        os.makedirs(path_letters)  # Создаем основную папку, если она не существует
    for leter in symbols:
        if not os.path.exists(path_letters+leter):
            os.makedirs(path_letters+leter)  # Создаем папку с именем символа, если она не существует
    if not os.path.exists(path_letters_etalon):
        os.makedirs(path_letters_etalon)  # Создаем основную папку, если она не существует
    for leter in symbols:
        if not os.path.exists(path_letters_etalon+leter):
            os.makedirs(path_letters_etalon+leter)

    start_time = time.time()  # Запускаем таймер для расчета оставшегося времени
    progress_text = st.empty()  # Элемент для отображения текста о прогрессе
    remaining_time_text = st.empty()  # Элемент для отображения времени

    if generate_by_letter:
        total_iterations = len(symbols) * num_generations
        progress_bar = tqdm(total=total_iterations, desc="Генерация изображений")
        progress_bar.update(1)

        for i in range(num_generations):
            for leter in symbols:
                img, etalon = image_imit.generate(leter, selected_font,
                                          scale_factor=np.random.uniform(scale_factor_range[0], scale_factor_range[1]),
                                          symbpl_angle = random.choice([-1, 1]) * np.random.uniform(symbpl_angle_range[0], symbpl_angle_range[1]),
                                          light_azdeg = np.random.uniform(light_angle_range[0], light_angle_range[1]),
                                          light_altdeg = np.random.uniform(light_angle_range[0], light_angle_range[1]),
                                          elev=random.choice([-1, 1]) * np.random.uniform(foto_angle_range[0], foto_angle_range[1]),
                                          azim=random.choice([-1, 1]) * np.random.uniform(foto_angle_range[0], foto_angle_range[1]),
                                          light_power=np.random.uniform(light_power_range[0], light_power_range[1]))

                letter_path = os.path.join(path_letters, leter)  # Use os.path.join for better path handling
                letter_path_etalon = os.path.join(path_letters_etalon, leter)

                filename = f"{leter}_{i}.png"
                filepath = os.path.join(letter_path, filename)
                filepath_etalon = os.path.join(letter_path_etalon, filename)
                j = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(letter_path, f"{leter}_{i}_{j}.png")
                    filepath_etalon = os.path.join(letter_path_etalon, f"{leter}_{i}_{j}.png")
                    j += 1
                try:
                    img.save(filepath)
                    etalon.save(filepath_etalon)
                except:
                    print('cannot save img ' + filepath)


                #img.save(os.path.join(path+leter, f"{leter}_{i}.png"))  # Сохраняем изображение в папке
                progress_text.text(f"Генерация {progress_bar.n}/{progress_bar.total}...")

                progress_bar.update(1)
                # Оставшееся время
                elapsed_time = time.time() - start_time
                remaining_time = total_iterations * elapsed_time / progress_bar.n - elapsed_time
                remaining_time_text.text(f"Оставшееся время: {remaining_time:.2f} секунд")
    else:
        if not os.path.exists(path_words):
            os.makedirs(path_words)  # Создаем основную папку, если она не существует
        if not os.path.exists(path_words_etalon):
            os.makedirs(path_words_etalon)  # Создаем основную папку, если она не существует

        total_iterations = num_generations
        progress_bar = tqdm(total=total_iterations, desc="Генерация изображений")
        progress_bar.update(1)

        for i in range(num_generations):
            leter = image_imit.get_random_cell_from_first_column()
            img, etalon = image_imit.generate(leter, selected_font,
                                      scale_factor=np.random.uniform(scale_factor_range[0], scale_factor_range[1]),
                                      symbpl_angle=random.choice([-1, 1]) * np.random.uniform(symbpl_angle_range[0], symbpl_angle_range[1]),
                                      light_azdeg=np.random.uniform(light_angle_range[0], light_angle_range[1]),
                                      light_altdeg=np.random.uniform(light_angle_range[0], light_angle_range[1]),
                                      elev=random.choice([-1, 1]) * np.random.uniform(foto_angle_range[0],
                                                                                      foto_angle_range[1]),
                                      azim=random.choice([-1, 1]) * np.random.uniform(foto_angle_range[0],
                                                                                      foto_angle_range[1]),
                                      light_power=np.random.uniform(light_power_range[0], light_power_range[1]))
            filename = f"{leter}_{i}.png"
            filepath = os.path.join(path_words, filename)
            filepath_etalon = os.path.join(path_words_etalon, filename)
            j = 1


            while os.path.exists(filepath):
                filepath = os.path.join(path_words, f"{leter}_{i}_{j}.png")
                filepath_etalon = os.path.join(path_words_etalon, f"{leter}_{i}_{j}.png")
                j += 1
            try:
                img.save(filepath)
                etalon.save(filepath_etalon)
            except:
                print('cannot save img ' + filepath)
            progress_text.text(f"Генерация {progress_bar.n}/{progress_bar.total}...")

            progress_bar.update(1)
            # Оставшееся время
            elapsed_time = time.time() - start_time
            remaining_time = total_iterations * elapsed_time / progress_bar.n - elapsed_time
            remaining_time_text.text(f"Оставшееся время: {remaining_time:.2f} секунд")

    progress_bar.close()
    st.success("Изображения успешно сгенерированы и сохранены в папках.")


