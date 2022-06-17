import streamlit as st
import numpy as np
import tensorflow as tf
import os, urllib
import librosa  # to extract speech features
import copy
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    # print(cv2.__version__)
    title = "Speech Emotion Analysis"
    st.title(title)
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition', 'view source code','view results')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('To try by yourself by adding a audio file .')
        #application()
        models_load_state = st.text('\n Loading models..')
        model = tf.keras.models.load_model('mymodel.h5')
        models_load_state.text('\n Models Loading..complete')
        upl(model)
    if selected_box == 'view source code':
        st.code(get_file_content_as_string('app.py'))
        # embed streamlit docs in a streamlit app
        st.markdown('lstm implementation source code''<a href="https://www.kaggle.com/code/roysonclausitdmello/fork-of-fork-of-fork-of-lstmser" target="_blank">...</a>', unsafe_allow_html=True)
        st.markdown('svm feature extraction source code''<a href="https://colab.research.google.com/drive/1XQ4ipDbASyMarNET1nYbUXbh-Dz162eV#scrollTo=4f8e89d1" target="_blank">...</a>', unsafe_allow_html=True)
        st.markdown('svm implementation source code''<a href="https://colab.research.google.com/drive/1XQ4ipDbASyMarNET1nYbUXbh-Dz162eV#scrollTo=4f8e89d1" target="_blank">...</a>', unsafe_allow_html=True)

    if selected_box == 'view results':
        displayImages()


def displayImages():
    st.text("Classification report of SER for RAVDESS and TESS using SVM")
    st.image("https://raw.githubusercontent.com/Royson609/Emo/main/images/classification_svm_ravdess.jpg", width=500,
             use_column_width=500)
    st.text("\n\n")
    st.text("Confusion matrix of SER for RAVDESS and TESS using SVM")
    st.image("https://raw.githubusercontent.com/Royson609/Emo/main/images/confusion_matrix_svm_ravdess.png", width=500,
             use_column_width=500)
    st.text("\n\n")
    st.text("Classification report of SER for RAVDESS and TESS using LSTM")
    st.image("https://raw.githubusercontent.com/Royson609/Emo/main/images/classification_lstm_ravdess.jpg", width=500,
             use_column_width=500)
    st.text("\n\n")
    st.text("\nConfusion matrix of SER for RAVDESS and TESS using LSTM")
    st.image("https://raw.githubusercontent.com/Royson609/Emo/main/images/confusion_matrix_lstm_ravdess.png", width=600,
             use_column_width=500)


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Royson609/Emo/main/app.py'
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")



def create_waveplot(data, sr):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio ', size=15)
    librosa.display.waveshow(data, sr=sr)
    return plt.gcf()

def create_spectrogram(data, sr):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio ', size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    return plt.gcf()


@st.cache(show_spinner=False)
def load_model():
    models_load_state = st.text('\n Loading models..')
    model = tf.keras.models.load_model('mymodel.h5')
    models_load_state.text('\n Models Loading..complete')
    return model


def upl(model):
    file_to_be_uploaded = st.file_uploader("Choose an audio...", type="wav")
    #path_in=file_to_be_uploaded.name

    y, sr = librosa.load(file_to_be_uploaded)

    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        st.pyplot(create_waveplot(y, sr))
        st.pyplot(create_spectrogram(y, sr))
        st.success('Emotion of the audio is  ' + predict(model, file_to_be_uploaded))

@st.cache(allow_output_mutation=True)
def application():
    model = load_model()





def extract_mfcc(wav_file_name):
    # This function extracts mfcc features and obtain the mean of each dimension
    # Input : path_to_wav_file
    # Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    return mfccs


def predict(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    print(emotions[np.argmax(predictions[0]) + 1])

    return emotions[np.argmax(predictions[0]) + 1]


if __name__ == "__main__":
    main()
