import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path
from api_key import assembly_api_key, elevenlabs_api_key
import os


def voice_to_voice(audio_file):
    # Print the value of audio_file for debugging
    print(f"Received audio file: {audio_file}")

    # Check if the file exists and is not empty
    if audio_file is None:
        raise gr.Error("No file was uploaded. Please upload a valid audio file.")
    
    if not os.path.isfile(audio_file) or os.path.getsize(audio_file) == 0:
        raise gr.Error("Uploaded file is empty or invalid. Please upload a valid audio file.")

    # Transcribe audio
    transcription_response = audio_transcription(audio_file)
    
    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    # Translate text
    hindi_translation, japanese_translation, korean_translation = text_translation(text)
    
    # Convert text to speech
    hindi_audio_path = text_to_speech(hindi_translation)
    jp_audio_path = text_to_speech(japanese_translation)
    ko_audio_path = text_to_speech(korean_translation)

    return hindi_audio_path, jp_audio_path, ko_audio_path


def audio_transcription(audio_file):
    aai.settings.api_key = assembly_api_key
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    return transcription


def text_translation(text):
    hindi_translator = Translator(from_lang="en", to_lang="hi")
    hindi_text = hindi_translator.translate(text)

    japanese_translator = Translator(from_lang="en", to_lang="ja")  
    japanese_text = japanese_translator.translate(text)

    korean_translator = Translator(from_lang="en", to_lang="ko")
    korean_text = korean_translator.translate(text)

    return hindi_text, japanese_text, korean_text


def text_to_speech(text):
    client = ElevenLabs(api_key=elevenlabs_api_key)
    
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice // voice assistant 
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",  # Change of model was done 
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Save in a specific directory with a timestamp
    save_dir = Path("generated_audio")
    save_dir.mkdir(exist_ok=True)
    save_file_path = save_dir / f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    return save_file_path


# Define the audio input
audio_input = gr.Audio(sources=["microphone"], type='filepath')

# Define the Gradio interface
demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Audio(label="Hindi"),
        gr.Audio(label="Japanese"),
        gr.Audio(label="Korean")
    ]
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
