import streamlit as st
import torch
from PIL import Image
from io import *
import glob
import os
import pandas as pd
import os
import shutil

cfg_model_path = "best_model.pt"
classes = ["Tất cả", "A1", "A2", "A3", "B4", "B5", "C6", "C7", "C8", "C9", "C10", "C11",
           "C12", "D13", "D14", "D15", "E16", "E17", "E18", "F19", "F20", "G21", "G22", "X", "Y"]


def imageInput(device, src):
    folder = r'runs/inference'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    if src == 'Upload your own data.':
        image_file = st.file_uploader(
            "Chọn ảnh tải lên", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            option = st.multiselect("Chọn class bạn muốn hiển thị", ["Tất cả", "A1", "A2", "A3", "B4", "B5", "C6",
                                                                     "C7", "C8", "C9", "C10", "C11", "C12", "D13", "D14", "D15", "E16", "E17", "E18", "F19", "F20", "G21", "G22", "X", "Y"], ["Tất cả"])
            submit = st.button("Dự đoán!")
            with col1:
                st.image(img, caption='Ảnh đã chọn', use_column_width='always')
            # ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data\\uploads', image_file.name)
            # outputpath = os.path.join('data\\outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            img_name = (imgpath.split('\\')[-1].split('.')[0])

            with col2:
                if image_file is not None and submit and 'Tất cả' in option:
                    os.system(
                        f'python tools/infer.py --yaml chromosome.yaml --weights {cfg_model_path} --source {imgpath} --save-txt')
                    pred = Image.open(f'runs/inference/exp/{img_name}.jpg')
                    st.image(pred, caption='Mô hình dự đoán')
                elif image_file is not None and submit and 'Tất cả' not in option:
                    options = [classes.index(i)-1 for i in option]
                    options = ' '.join(map(str, options))
                    print(options)
                    os.system(
                        f'python tools/infer.py --yaml chromosome.yaml --weights {cfg_model_path} --source {imgpath} --save-txt --classes {options}')
                    pred = Image.open(f'runs/inference/exp/{img_name}.jpg')
                    st.image(pred, caption='Mô hình dự đoán')

            if submit:
                st.divider()

                df = pd.read_csv(
                    f'runs/inference/exp/labels/{img_name}.txt', sep=" ",index_col=False, header=None)
                df.columns = ['0', '1', '2', '3', '4', '5']
                df = df[['0', '5']]
                df.columns = ["Class", "Confidence Score"]
                df['Class'] = df['Class'].apply(lambda x: classes[x+1])
                df['Confidence Score'] = df['Confidence Score'].apply(
                    lambda x: round(x*100, 2))

                df['Class'] = sorted(df['Class'], key=custom_sort_key)

                st.table(df)

                st.divider()
                if 'Tất cả' in option:
                    display_text = abnormaly_detection(df)
                    st.text(display_text)

    elif src == 'From test set.':
        # Image selector slider
        imgpath = glob.glob('data/test/*')
        imgsel = st.slider('Chọn ảnh từ tập test.',
                           min_value=1, max_value=len(imgpath), step=1)
        image_file = imgpath[imgsel-1]
        option = st.multiselect("Chọn class bạn muốn hiển thị", ["Tất cả", "A1", "A2", "A3", "B4", "B5", "C6",
                                                                 "C7", "C8", "C9", "C10", "C11", "C12", "D13",
                                                                 "D14", "D15", "E16", "E17", "E18", "F19", "F20",
                                                                 "G21", "G22", "X", "Y"],
                                ["Tất cả"])
        submit = st.button("Dự đoán!")
        col1, col2 = st.columns(2)
        img_name = (image_file.split('\\')[-1].split('.')[0])
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Ảnh đã chọn', use_column_width='always')
        with col2:
            if image_file is not None and submit and 'Tất cả' in option:
                os.system(
                    f'python tools/infer.py --yaml chromosome.yaml --weights {cfg_model_path} --source {image_file} --save-txt')
                pred = Image.open(f'runs/inference/exp/{img_name}.jpg')
                st.image(pred, caption='Mô hình dự đoán')

            elif image_file is not None and submit and 'Tất cả' not in option:
                options = [classes.index(i)-1 for i in option]
                options = ' '.join(map(str, options))
                os.system(
                    f'python tools/infer.py --yaml chromosome.yaml --weights {cfg_model_path} --source {image_file} --save-txt --classes {options}')
                pred = Image.open(f'runs/inference/exp/{img_name}.jpg')
                st.image(pred, caption='Mô hình dự đoán')
        if submit:
            st.divider()
            df = pd.read_csv(
                f'runs/inference/exp/labels/{img_name}.txt', sep=" ",index_col=False, header=None)
            df.columns = ['0', '1', '2', '3', '4', '5']
            df = df[['0', '5']]
            df.columns = ["Class", "Confidence Score"]
            df['Class'] = df['Class'].apply(lambda x: classes[x+1])
            df['Confidence Score'] = df['Confidence Score'].apply(
                lambda x: round(x*100, 2))

            df['Class'] = sorted(df['Class'], key=custom_sort_key)
            st.table(df)
            st.divider()

            if 'Tất cả' in option:
                display_text = abnormaly_detection(df)
                st.text(display_text)


def custom_sort_key(x):
    # Extract alphabetical characters and numerical characters
    alpha = ''.join([c for c in x if c.isalpha()])
    num = ''.join([c for c in x if c.isdigit()])

    # Return a tuple with the alphabetical characters and numerical characters
    return alpha, int(num) if num else float('inf')


def abnormaly_detection(df):
    total_chromsomes = len(df)
    ch13 = (df['Class'] == 'D13').sum()
    ch18 = (df['Class'] == 'E18').sum()
    ch21 = (df['Class'] == 'G21').sum()
    ch23 = (df['Class'] == 'X').sum()
    ch24 = (df['Class'] == 'Y').sum()
    text = ''
    if total_chromsomes == 47:
        if ch13 >= 3:
            text = 'Trisomy 13'
        if ch18 >= 3:
            text = 'Trisomy 18'
        if ch21 >= 3:
            text = 'Trisomy 21'
        if ch23 >= 2 and ch24 >= 1:
            text = 'Trisomy XXY'
    elif total_chromsomes == 45 and ch23 == 1 and ch24 == 0:
        text = 'Monosomy X'
    elif total_chromsomes == 46:
        text = 'Normal'
    else:
        text = 'Abnormal'
    return text


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", [
                               'From test set.', 'Upload your own data.'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('🧬Nhận diện nhiễm sắc thể🧬')
    st.divider()
    imageInput(deviceoption, datasrc)


if __name__ == '__main__':
    main()
