# generate_beep.py
import wave
import struct
import math

file_path = "alert.wav"

duration = 0.5      # seconds
freq = 1000         # Hz
volume = 0.8
sample_rate = 44100

num_samples = int(duration * sample_rate)

with wave.open(file_path, 'w') as wav_file:
    wav_file.setnchannels(1)       
    wav_file.setsampwidth(2)       
    wav_file.setframerate(sample_rate)

    for i in range(num_samples):
        sample = volume * math.sin(2 * math.pi * freq * (i / sample_rate))
        wav_file.writeframes(struct.pack('<h', int(sample * 32767)))

print("alert.wav created successfully!")
