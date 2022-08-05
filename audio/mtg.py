from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from einops import rearrange

from audio.util import load_normalizers


def generator_fn(mtg_root_path: Path, split: str, sampling_rate, duration, n_fft, hop_length, normalizers):
    """
    Generates a dataset of spectrograms from the MTG dataset.

    Args:
        mtg_root_path: Path to the root directory of the MTG dataset.
        split: The split to use (train/valid)
        sampling_rate: The sampling rate of the audio in Hz.
        duration: The duration of the final audio sample in seconds.
        n_fft: The length of the window signal for the fourier transformation.
        hop_length: The number of samples between each window.
        normalizers: Dictionary with real and imag normalizers to use for the spectrograms.
    """
    tracks_file = Path(mtg_root_path).joinpath(f"{split}.tsv")
    tracks = pd.read_csv(tracks_file, sep="\t")

    def generator():
        for i, track in tracks.iterrows():
            audio_path = mtg_root_path.joinpath(f"opus/{split}/{track.chunk_nr}/{track.id}.opus")
            if not audio_path.exists():
                continue
            audio_array = librosa.load(audio_path, sr=sampling_rate, duration=duration + 10)[0]
            audio_array = librosa.effects.trim(audio_array)[0]
            audio_array = audio_array[:int(sampling_rate * duration)]
            spec = spectrogram(audio_array, n_fft, hop_length)
            spec = spec[:-1, :]  # crop the highest frequency
            spec = normalize(spec, normalizers)
            yield {"label": int(track.id), "image": spec}

    return generator


def spectrogram(audio_array, n_fft, hop_length):
    """
    Computes the spectrogram of the audio array.
    """
    X = librosa.stft(np.asarray(audio_array), n_fft=n_fft, hop_length=hop_length)
    X = np.stack([X.real, X.imag], axis=-1).astype("float32")
    return X


def normalize(X, normalizers):
    X_real = normalizers["real"].transform(rearrange(X[:, :, 0], "f t -> t f"))
    X_imag = normalizers["imag"].transform(rearrange(X[:, :, 1], "f t -> t f"))

    X_real = rearrange(X_real, "t f -> f t")
    X_imag = rearrange(X_imag, "t f -> f t")

    X = np.stack([X_real, X_imag], axis=-1).astype("float32")
    return X


def get_mtg_dataset(path, split, sampling_rate, duration, n_fft, hop_length, normalizers) -> tf.data.Dataset:
    """
    Returns a dataset of spectrograms from the MTG dataset.

    Args:
        path: Path to the root directory of the MTG dataset.
        split: The split to use (train/valid)
        sampling_rate: The sampling rate of the audio in Hz.
        duration: The duration of the final audio sample in seconds.
        n_fft: The length of the window signal for the fourier transformation.
        hop_length: The number of samples between each window.
        normalizers: Dictionary with real and imag normalizers to use for the spectrograms.
    """
    time_bins = int(np.ceil((duration * sampling_rate) / hop_length))
    return tf.data.Dataset.from_generator(generator_fn(path, split ,sampling_rate, duration, n_fft, hop_length, normalizers),
                                          output_signature={
                                              "label": tf.TensorSpec(shape=[], dtype=tf.int32),
                                              "image": tf.TensorSpec(shape=[n_fft // 2, time_bins, 2])
                                          })


if __name__ == '__main__':
    audio_normalizers = load_normalizers(Path(__file__).parent.joinpath("audio_normalizers.pickel"))
    ds = get_mtg_dataset(Path("/Volumes/Black T5/dataset/mtg-jamendo"), "train", sampling_rate=44100, duration=10, n_fft=1024, hop_length=512, normalizers=audio_normalizers)

    ds = ds.batch(8, drop_remainder=True)
    for element in ds.as_numpy_iterator():
        print(element["label"].shape, element["image"].shape)
        break
