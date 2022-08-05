from functools import partial
from pathlib import Path

import os
import librosa
import numpy as np
import pandas as pd
import soundfile
import tensorflow as tf
from einops import rearrange

from audio.util import load_normalizers


def generator_fn(mtg_root_path: Path, split: str, sampling_rate, duration, n_fft, hop_length, normalizers, genre=None):
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
        genre: Will return only tracks that have this genre under genres
    """
    tracks_file = Path(mtg_root_path).joinpath(f"{split}.tsv")
    tracks = pd.read_csv(tracks_file, sep="\t")
    print(f"Found {len(tracks)} tracks in {tracks_file}")
    if genre is not None:
        tracks = tracks[tracks["genres"].apply(lambda gs: genre in eval(gs))]
        print(f"Filtering on genre '{genre}'. {len(tracks)} tracks left.")

    def generator():
        for i, track in tracks.iterrows():
            audio_path = mtg_root_path.joinpath(f"opus/{track.id}.opus")
            if not audio_path.exists():
                print("Skipping", audio_path, "(not found)")
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


def get_mtg_dataset(path, split, sampling_rate, duration, n_fft, hop_length, normalizers, genre: str=None) -> tf.data.Dataset:
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
    return tf.data.Dataset.from_generator(generator_fn(path, split ,sampling_rate, duration, n_fft, hop_length, normalizers, genre=genre),
                                          output_signature={
                                              "label": tf.TensorSpec(shape=[], dtype=tf.int32),
                                              "image": tf.TensorSpec(shape=[n_fft // 2, time_bins, 2])
                                          })


if __name__ == '__main__':

    normalizers = load_normalizers(Path(__file__).parent.joinpath("audio_normalizers.pickel"))
    num_epochs = None
    shuffle_buffer_size = 50
    batch_size = 8
    prefetch_size = tf.data.AUTOTUNE

    def prepare_dataset(ds: tf.data.Dataset):
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    mtg_dataset = partial(get_mtg_dataset,
                            path=Path(os.environ.get("MTG_DATASET_PATH")),
                            genre="lofi",
                            sampling_rate=22050,
                            duration=10,
                            n_fft=1024,
                            hop_length=431,
                            normalizers=normalizers
                            )
    print("Loading MTG dataset")
    train_ds = prepare_dataset(mtg_dataset(split="train"))

    print(train_ds)
    for el in iter(train_ds):
        print(el)
        break