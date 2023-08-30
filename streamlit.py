import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor
import os

st.title("Text-to-Speech Synthesis")

# Sidebar
model_selection = st.sidebar.selectbox("Select Model", [
    "Tacotron2 + MelGAN",
    "Tacotron2 + MelGAN-STFT",
    "Tacotron2 + MB-MelGAN",
    "FastSpeech + MB-MelGAN",
    "FastSpeech + MelGAN-STFT",
    "FastSpeech + MelGAN",
    "FastSpeech2 + MB-MelGAN",
    "FastSpeech2 + MelGAN-STFT",
    "FastSpeech2 + MelGAN"
])

input_text = st.text_area("Enter Text", value="Bill got in the habit of asking himself “Is that thought true?” And if he wasn’t absolutely certain it was, he just let it go.")

# Load models and configurations
@st.cache(allow_output_mutation=True)
def load_models():
    processor = AutoProcessor.from_pretrained("https://1drv.ms/u/s!AkBZ3UdLQfXChQBZLG9Rdp-92xXU?e=d9u7iQ")
    config_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChQSEa87uq4gWv0Al?e=d0X3Ih"
    melgan_stft_config = AutoConfig.from_pretrained(config_path)
    
    tacotron2_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChQBZLG9Rdp-92xXU?e=d9u7iQ"
    tacotron2 = TFAutoModel.from_pretrained(tacotron2_path, config=tacotron2_path+"/config.json")
    
    fastspeech_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChQG8x1ycUplu6ERM?e=NgUYH4"
    fastspeech = TFAutoModel.from_pretrained(fastspeech_path, config=fastspeech_path+"/config.json")
    
    fastspeech2_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChRNs1N746J4FbDV_?e=6ppTPO"
    fastspeech2 = TFAutoModel.from_pretrained(fastspeech2_path, config=fastspeech2_path+"/config.json")
    
    melgan_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChH0LD1It_1i3Rrxt?e=sR89Cd"
    melgan = TFAutoModel.from_pretrained(melgan_path, config=melgan_path+"/config.json")
    
    melgan_stft_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChQVJmfi0lYuXO2cE?e=yUKD2d"
    melgan_stft = TFAutoModel.from_pretrained(melgan_stft_config, pretrained_path=melgan_stft_path, config=melgan_stft_config)
    
    mb_melgan_path = "https://1drv.ms/u/s!AkBZ3UdLQfXChH9GExHjhtHhx-sU?e=YGFXXv"
    mb_melgan = TFAutoModel.from_pretrained(mb_melgan_path, config=mb_melgan_path+"/config.json")
    
    return processor, tacotron2, fastspeech, fastspeech2, melgan, melgan_stft, mb_melgan

processor, tacotron2, fastspeech, fastspeech2, melgan, melgan_stft, mb_melgan = load_models()

def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
    input_ids = processor.text_to_sequence(input_text)

    if text2mel_name == "Tacotron2 + MelGAN":
        _, mel_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
    elif text2mel_name == "FastSpeech + MB-MelGAN":
        _, mel_outputs, _ = text2mel_model.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    elif text2mel_name == "FastSpeech2 + MelGAN":
        _, mel_outputs, _, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )
    else:
        raise ValueError("Invalid text2mel_name selected")

    if vocoder_name == "Tacotron2 + MelGAN":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    elif vocoder_name == "Tacotron2 + MelGAN-STFT":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    elif vocoder_name == "Tacotron2 + MB-MelGAN":
        audio = vocoder_model(mel_outputs)[0, :, 0]
    else:
        raise ValueError("Invalid vocoder_name selected")

    return mel_outputs.numpy(), audio.numpy()

if st.button("Synthesize"):
    # Perform synthesis based on selected model
    if model_selection == "Tacotron2 + MelGAN":
        mel_outputs, audio = do_synthesis(input_text, tacotron2, melgan, "Tacotron2 + MelGAN", "Tacotron2 + MelGAN")
    elif model_selection == "Tacotron2 + MelGAN-STFT":
        mel_outputs, audio = do_synthesis(input_text, tacotron2, melgan_stft, "Tacotron2 + MelGAN", "Tacotron2 + MelGAN-STFT")
    elif model_selection == "Tacotron2 + MB-MelGAN":
        mel_outputs, audio = do_synthesis(input_text, tacotron2, mb_melgan, "Tacotron2 + MelGAN", "Tacotron2 + MB-MelGAN")
    elif model_selection == "FastSpeech + MB-MelGAN":
        mel_outputs, audio = do_synthesis(input_text, fastspeech, mb_melgan, "FastSpeech + MB-MelGAN", "FastSpeech + MB-MelGAN")
    elif model_selection == "FastSpeech + MelGAN-STFT":
        mel_outputs, audio = do_synthesis(input_text, fastspeech, melgan_stft, "FastSpeech + MelGAN", "FastSpeech + MelGAN-STFT")
    elif model_selection == "FastSpeech + MelGAN":
        mel_outputs, audio = do_synthesis(input_text, fastspeech, melgan, "FastSpeech + MelGAN", "FastSpeech + MelGAN")
    elif model_selection == "FastSpeech2 + MB-MelGAN":
        mel_outputs, audio = do_synthesis(input_text, fastspeech2, mb_melgan, "FastSpeech2 + MB-MelGAN", "FastSpeech2 + MB-MelGAN")
    elif model_selection == "FastSpeech2 + MelGAN-STFT":
        mel_outputs, audio = do_synthesis(input_text, fastspeech2, melgan_stft, "FastSpeech2 + MelGAN", "FastSpeech2 + MelGAN-STFT")
    elif model_selection == "FastSpeech2 + MelGAN":
        mel_outputs, audio = do_synthesis(input_text, fastspeech2, melgan, "FastSpeech2 + MelGAN", "FastSpeech2 + MelGAN")
    else:
        raise ValueError("Invalid model_selection")

    # Display spectrogram
    fig, ax = plt.subplots()
    ax.imshow(mel_outputs[0].T, aspect="auto", origin="lower")
    ax.set_title("Generated Spectrogram")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bins")
    st.pyplot(fig)

    # Play audio
    st.audio(audio, format="audio/wav")
