import moviepy as mp
import speech_recognition as sr
import os

def extract_audio_from_video(video_path, audio_path):
  video = mp.VideoFileClip(video_path)
  video.audio.write_audiofile(audio_path)

def transcribe_audio_to_text(audio_path, text_output_path):
  recognizer = sr.Recognizer()

  with sr.AudioFile(audio_path) as source:
    # lê todo o áudio do arquivo
    audio = recognizer.record(source)

    try:
      # usa o serviço de reconhecimento de fala do Google
      text = recognizer.recognize_google(audio, language='pt-BR')
      print('Transcrição: ', text)

      # salva a transcrição em um arquivo de texto
      with open(text_output_path, 'w', encoding='utf-8') as file:
        file.write(text)

    except sr.UnknownValueError:
      print("Google Speech Recognition não conseguiu entender o áudio")
    except sr.RequestError as e:
      print("Erro ao solicitar resultados do serviço de reconhecimento de fala do Google; {0}".format(e))

def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  video_path = os.path.join(script_dir, 'video1.mp4') # vídeo de entrada
  audio_path = os.path.join(script_dir, 'audio1.wav')
  text_output_path = os.path.join(script_dir, 'transcricao-video.txt')

  extract_audio_from_video(video_path, audio_path)
  transcribe_audio_to_text(audio_path, text_output_path)

if __name__ == "__main__":
  main()
