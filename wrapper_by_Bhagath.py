from pathlib import Path
import encoder.inference as encoderInference
import encoder.audio as encoderAudio
import synthesizer.inference as SynthesizerInference
import vocoder.inference as vocoderInference
import numpy as np
import librosa
import warnings
from IPython.display import Audio
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

encoderInference.load_model(Path("saved_models/default/encoder.pt"))
synObject = SynthesizerInference.Synthesizer(Path("saved_models/default/synthesizer.pt"))
vocoderInference.load_model(Path("saved_models/default/vocoder.pt"))

def wrapperFunc(fileName, startTime, endTime, text):
    wavData, samplingRate = librosa.load(fileName, offset=startTime, duration=endTime)
    # print(len(wavData), samplingRate)
    embedding = encoderInference.embed_utterance(encoderAudio.preprocess_wav(fpath_or_wav = wavData, source_sr = samplingRate))
    # print(embedding)
    print("Synthesizing Text to Audio.............")
    specs = synObject.synthesize_spectrograms([text], [embedding])
    generated_wav = vocoderInference.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synObject.sample_rate), mode="constant")
    print(generated_wav)
    audio = Audio(generated_wav, rate=synObject.sample_rate, autoplay=True)
    outputFileName = 'NewAudio' + '_OF_' + fileName[:-4] + '.wav'
    with open(outputFileName, 'wb') as f:
        f.write(audio.data)

# OPTIONAL: convert the audio file from .mp3 to .wav using the command "ffmpeg -i Interactly_lady_voice_05.mp3 -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav Interactly_lady_voice_05.wav"
text = "Hi, welcome to Interactly, a no coding interactive video creation platform to create the personalized video experiences."
fileName = 'PRIYANKA_CHOPRA_ Be Fearless05.mp3'
startTime=0
endTime=10
wrapperFunc(fileName, startTime, endTime, text)