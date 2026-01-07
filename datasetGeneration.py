#  Synthetic audio dataset generation
# Simulates three musical instruments using distinct frequency and harmonic patterns.
# Each instrument class contains 100 audio samples of fixed duration (3 seconds).

import numpy as np
import soundfile as sf
import os

sr = 22050
duration = 3
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

def flute():
    return 0.5 * np.sin(2 * np.pi * 1000 * t)

def violin():
    return sum((1/n) * np.sin(2*np.pi*440*n*t) for n in range(1, 6))

def trumpet():
    return sum((1/n) * np.sin(2*np.pi*600*n*t) for n in range(1, 4))

generators = {
    "flute": flute,
    "violin": violin,
    "trumpet": trumpet
}

os.makedirs("synthetic_audio", exist_ok=True)

for label, gen in generators.items():
    os.makedirs(f"synthetic_audio/{label}", exist_ok=True)
    for i in range(100):
        audio = gen()
        sf.write(f"synthetic_audio/{label}/{label}_{i}.wav", audio, sr)

print("Synthetic dataset created successfully")
