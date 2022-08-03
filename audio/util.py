import _pickle
from typing import Dict, List

import librosa
import numpy as np
from einops import rearrange
from functional import seq
from sklearn.preprocessing import QuantileTransformer


def load_normalizers(path):
    """Loads the normalizers serialized under the given path

    Args:
        path: Path to serialized quantile transformer normalizers

    Returns:
        (dict) real and imag quantile normalizers
    """
    with open(path, "rb") as f:
        params = _pickle.load(f)

    def load_normalizer(params, key):
        normalizer = QuantileTransformer()
        normalizer.set_params(**params[key]["params"])
        normalizer.quantiles_ = params[key]["quantiles"]
        normalizer.references_ = params[key]["references"]
        return normalizer

    return {
        "real": load_normalizer(params, "real"),
        "imag": load_normalizer(params, "imag")
    }


def filter_func(samples: Dict[str, List], genre="lofi") -> List[bool]:
    return [genre in x for x in samples["genres"]]


def load_audio(samples, sampling_rate=22050, remove_silence=True, duration=10):
    """

    Args:
        samples: Containing key audio, cast to Audio(decode=False)
        sampling_rate: Sampling rate
        remove_silence: (bool) Whether to remove leading and trailing silence
        duration: (numeric) Only load up to this much audio (in seconds), return everything if None

    Returns:
        Dict with "audio_array" list of decoded audio signal (amplitude)
    """

    def take_first_few_seconds(y):
        return y[:int(sampling_rate * duration)]

    x = (seq([x["path"] for x in samples["audio"]])
         .map(lambda x: librosa.load(x, duration=duration + 10, sr=sampling_rate)[0])
         .map(lambda x: librosa.effects.trim(x)[0] if remove_silence else x)
         .map(take_first_few_seconds)
         ).to_list()

    print("load", type(x))
    return {
        "audio_array": x
    }


def create_spectrogram(samples, n_fft=1024, hop_length=512):
    """

    Args:
        samples: (dict) samples containing key "audio_array"
        n_fft: number of samples for fourier transformation
        hop_length: hop length for stft

    Returns:
        Dict with "audio_spectrogram": List of spectrograms (samples, freq, time, (real/imag)) computed from "audio_array"

    Note:
        Since pyarrow does not support complex datatypes the real and imaginary parts of the spectogram are split and
        differentiated in the last dimension
    """
    Y = np.stack(samples["audio_array"], axis=0)
    X = librosa.stft(Y, n_fft=n_fft, hop_length=hop_length)
    X = np.stack([X.real, X.imag], axis=3).astype("float32")

    print("spec", X.shape, X.dtype)
    return {
        "audio_spectrogram": X
    }


def crop_spectrogram(samples):
    X = np.asarray(samples["audio_spectrogram"])
    print("crop", X.shape)
    return {
        # X = (batch, freq, time, channel)
        "audio_spectrogram": X[:, :-1, :, :]
    }


def normalize_spectrogram(samples, normalizers):
    """

    Args:
        samples:
        normalizers: fitted scipy transformation, usually Quantile

    Returns:

    """
    X = np.asarray(samples["audio_spectrogram"])

    X_real = normalizers["real"].transform(rearrange(X[:, :, :, 0], "n f t -> (n t) f"))
    X_imag = normalizers["imag"].transform(rearrange(X[:, :, :, 1], "n f t -> (n t) f"))

    X_real = rearrange(X_real, "(n t) f -> n f t", n=X.shape[0])
    X_imag = rearrange(X_imag, "(n t) f -> n f t", n=X.shape[0])

    X = np.stack([X_real, X_imag], axis=3).astype("float32")

    print("norm", X.shape, X.dtype)
    return {
        "audio_spectrogram": X
    }


def invert_normalization(samples, normalizers):
    Xs_real = normalizers["real"].inverse_transform(rearrange(samples[:, :, :, 0], "n f t -> (n t) f"))
    Xs_imag = normalizers["imag"].inverse_transform(rearrange(samples[:, :, :, 1], "n f t -> (n t) f"))

    Xs_real = rearrange(Xs_real, "(n t) f -> n f t", n=samples.shape[0])
    Xs_imag = rearrange(Xs_imag, "(n t) f -> n f t", n=samples.shape[0])
    return np.stack([Xs_real, Xs_imag], axis=3)


def invert_spectrogram(samples, n_fft, hop_length):
    samples = samples[:, :, :, 0] + 1j * samples[:, :, :, 1]
    Ys = librosa.istft(samples, n_fft=n_fft, hop_length=hop_length)
    return Ys
