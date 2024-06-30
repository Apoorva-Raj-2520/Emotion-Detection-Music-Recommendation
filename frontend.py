import streamlit as st
from PIL import Image
import subprocess,threading , pickle, vlc
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as keras_image

file_path = ""
song_thread = None

def load_data():
    return pd.read_excel('E:/6th_sem_stuff/tdl/project/spotify_playlist_tracks.xlsx', engine='openpyxl')

def Emotion(image_path, model_path='<Enter Model Name>'):
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the image
    img = keras_image.load_img(image_path, target_size=(46, 46), color_mode='grayscale')
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the emotion
    emotion_labels = ['Happy', 'Angry', 'Fear', 'Neutral', 'Sad', 'Disgust', 'Surprise']
    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    predicted_label = emotion_labels[predicted_label_index]

    return predicted_label


# def load_model_file(file_path):
#     model = tf.keras.models.load_model(file_path)
#     return model


# def Emotion(image = None):
    # st.image(image, caption='Uploaded Image', use_column_width=True)
    # model = load_model_file(file_path)
    # output = model.predict(image)
    # return output


def play_song(song_name):
    # global song_thread
    # def song_player():
        command = ['yt-dlp', '-x', '--audio-format', 'best', '--get-url', 'ytsearch1:' + song_name]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        url = result.stdout.strip()
        if not url:
            st.error("Could not retrieve song.")
            return

        player = vlc.MediaPlayer(url)
        player.play()
        # try:
        #     print("Playback started. Press ESC to stop.")
        #     while True:
        #         time.sleep(1)
        #         if keyboard.is_pressed('esc'):
        #             # print("ESC pressed. Stopping playback.")
        #             break
        #         if not player.is_playing():
        #             # print("Song finished playing.")
        #             break
        # except Exception as e:
        #     print("Error during playback:", e)
        # finally:
        #     player.stop()
        #     # print("Playback stopped.")
        
    # if song_thread and song_thread.is_alive():
    #     st.error("A song is already playing. Please stop the current song before playing another one.")
    # else:
    #     song_thread = threading.Thread(target=song_thread)
    #     song_thread.start()
    #     st.write("Playing")


def stop_song():
    global song_thread
    if song_thread is not None and song_thread.is_alive():
        song_thread.join(timeout=1)
        if song_thread.is_alive():
            st.warning("Failed to stop the song immediately.")
        else:
            st.success("Playback stopped.")
            song_thread = None
    else:
        st.error("No song is currently playing.")


def find_song_by_emotion(emotion, data):
    data.columns = data.columns.str.strip()
    filtered_songs = data[data['emotion'].str.lower() == emotion.lower()]

    if not filtered_songs.empty:
        SongName = filtered_songs.sample(n=1).iloc[0]
        return f"{SongName['track name']} by {SongName['artist']}",f"{SongName['shareUrl']}",f"{SongName['track id']}"
    return None



def main():
    st.title("Emotion Based Music Player")
    songs_data = load_data()
    # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption='Original Image')

    #     emo = Emotion('''image''') 
    song_name, spotify_url, yt_url = find_song_by_emotion("Happy", songs_data)
    if song_name is not None:
        st.write(f"Song Selected: {song_name}")
        if spotify_url:
            st.markdown(f"[Play on Spotify]({spotify_url})", unsafe_allow_html=True)
        if yt_url:
            st.markdown(f"[Play on YouTube](https://www.youtube.com/watch?v={yt_url})", unsafe_allow_html=True)
        if st.button("Play Now"):
            play_song(song_name)
        #     if st.button('Stop Song'):
        #         stop_song()

if __name__ == "__main__":
    main()
