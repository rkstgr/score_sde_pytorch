import os
from typing import Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms
import torchaudio.functional as T
from einops import rearrange
from pathlib import Path

from torchaudio.transforms import InverseSpectrogram


def load_normalizers(path=Path(__file__).parent / "audio_normalizers.pickel"):
    """Loads the normalizers serialized under the given path

    Args:
        path: Path to serialized quantile transformer normalizers

    Returns:
        (dict) real and imag quantile normalizers
    """
    import _pickle
    from sklearn.preprocessing import QuantileTransformer

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


def _load_tracks(track_file: Path, genres: Union[str, List[str], None]) -> pd.DataFrame:
    df = pd.read_csv(track_file, sep="\t")
    print("Loaded {} tracks".format(len(df)))
    df.genres = df.genres.apply(lambda x: eval(x))
    if genres:
        if isinstance(genres, str):
            df = df[df.genres.apply(lambda x: genres in x)]
        elif isinstance(genres, list):
            df = df[df.genres.apply(lambda x: any(g in x for g in genres))]
        else:
            raise ValueError("genres must be str or list: {}".format(genres))
        print("Filtered to {} tracks, for genres '{}'".format(len(df), genres))

    return df[["id", "genres"]]


class MtgOpusDataset(Dataset):
    def __init__(self,
                 track_file: Path,
                 opus_dir: Path,
                 genres: Union[str, List[str], None],
                 sampling_rate: int = 22050,
                 duration: float = 10.0,
                 remove_silence: bool = True,
                 transform=None,
                 target_transform=None
                 ):
        self.track_file = track_file
        self.opus_dir = opus_dir
        self.genres = genres
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.remove_silence = remove_silence
        self.transform = transform
        self.target_transform = target_transform

        torchaudio.set_audio_backend("sox_io")
        self.tracks = _load_tracks(track_file, genres)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        _id = self.tracks.iloc[idx, 0]
        track_path = self.opus_dir / f"{_id}.opus"
        track_orig, sample_rate = torchaudio.load(track_path,
                                                  num_frames=int((self.duration + 10) * 48000))
        track_orig = torch.mean(track_orig, dim=0)  # stereo to mono
        track = T.resample(track_orig, orig_freq=sample_rate, new_freq=self.sampling_rate)

        if self.remove_silence:
            nonzero = torch.nonzero(torch.abs(track) > 0.01)
            if torch.numel(nonzero) > 0:
                nonzero_index = nonzero[0][0]
                nonzero_index = min(nonzero_index, len(track) - int(self.duration * self.sampling_rate))
                track = track[nonzero_index:nonzero_index + int(self.duration * self.sampling_rate)]

        genres = self.tracks.iloc[idx, 1]
        label = genres[0] if len(genres) > 0 else "nothing"
        if self.transform:
            track = self.transform(track)
        if self.target_transform:
            label = self.target_transform(label)
        return {
            "track": track,
            "track_id": _id,
            "genre": label
        }


class StackRealImag(torch.nn.Module):
    @staticmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x.real, x.imag], dim=-1)


class NormalizeRealImag(torch.nn.Module):
    def __init__(self, normalizers):
        super().__init__()
        self.normalizers = normalizers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.stack([
            rearrange(self.normalizers["real"].transform(rearrange(x.real, "f t -> t f")), "t f -> f t"),
            rearrange(self.normalizers["imag"].transform(rearrange(x.imag, "f t -> t f")), "t f -> f t")
        ], axis=-1))[:-1, :, :]


class InverseNormalizeRealImag(torch.nn.Module):
    def __init__(self, normalizers):
        super().__init__()
        self.normalizers = normalizers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])], dim=1)
        re = rearrange(self.normalizers["real"].inverse_transform(rearrange(x[:, :, :, 0], "n f t -> (n t) f")), "(n t) f -> n f t", n=x.shape[0])
        im = rearrange(self.normalizers["imag"].inverse_transform(rearrange(x[:, :, :, 1], "n f t -> (n t) f")), "(n t) f -> n f t", n=x.shape[0])
        return torch.from_numpy(re + 1j * im)


class InverseSamples(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformation = nn.Sequential(
            FeaturesToComplex(
                ref=config.data.transforms.ref,
                amin=config.data.transforms.amin,
                top_db=config.data.transforms.top_db,
                sigmoid_temp=config.data.transforms.sigmoid_temp,
            ),
            InverseSpectrogram(n_fft=config.data.n_fft,
                               win_length=config.data.n_fft,
                               hop_length=config.data.hop_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformation(x)


class ComplexToFeatures(torch.nn.Module):
    def __init__(self, ref=1.0, amin=1e-05, top_db=80.0, sigmoid_temp=15.0):
        super().__init__()
        # ref = 1.0, amin = 1e-05, top_db = 80.0
        self.ref = torch.as_tensor(ref)
        self.amin = torch.as_tensor(amin)
        self.top_db = torch.as_tensor(top_db)
        self.sigmoid_temp = sigmoid_temp

    def magnitude_to_db(self, magnitude):
        log_spec = 10.0 * torch.log10(torch.maximum(self.amin, magnitude ** 2))
        log_spec -= 10.0 * torch.log10(torch.maximum(self.amin, self.ref))
        log_spec = torch.maximum(log_spec, torch.as_tensor(torch.max(log_spec) - self.top_db))
        return log_spec

    def sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x / self.sigmoid_temp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        x_angle = (torch.angle(x) + torch.pi) / (2 * torch.pi)  # normalized to [0, 1]
        db = self.magnitude_to_db(magnitude)
        db_norm = self.sigmoid(db)
        feat = torch.stack([db_norm, x_angle], dim=-1)
        # remove the last dimension (freq)
        if feat.dim() == 3:
            return feat[:-1, :, :]
        elif feat.dim() == 4:
            return feat[:, :-1, :, :]
        else:
            raise ValueError(f"Expected 3 or 4 dimensions, got {feat.dim()}")


class FeaturesToComplex(torch.nn.Module):
    def __init__(self, ref=1.0, amin=1e-05, top_db=80.0, sigmoid_temp=15.0):
        super().__init__()
        # ref = 1.0, amin = 1e-05, top_db = 80.0
        self.ref = torch.as_tensor(ref)
        self.amin = torch.as_tensor(amin)
        self.top_db = torch.as_tensor(top_db)
        self.sigmoid_temp = sigmoid_temp

    def sigmoid_inverse(self, db_norm):
        return torch.log(db_norm / (1 - db_norm)) * self.sigmoid_temp

    def db_to_magnitude(self, db):
        return self.ref * torch.pow(10.0, 0.1 * db)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add zeros to the first dimension (freq)
        if x.dim() == 3:
            x = torch.cat([x, torch.zeros(1, x.shape[1], x.shape[2])], dim=0)
        elif x.dim() == 4:
            x = torch.cat([x, torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])], dim=1)
        else:
            raise ValueError(f"Expected 3 or 4 dimensions, got {x.dim()}")
        db_norm = x[:, :, 0]
        x_angle_norm = x[:, :, 1] - 0.5
        angle = 2 * torch.pi * x_angle_norm
        db = self.sigmoid_inverse(db_norm)
        S = self.db_to_magnitude(db)
        S = torch.sqrt(S)
        return S * torch.cos(angle) + 1j * S * torch.sin(angle)


def get_mtg_dataset(split, config):
    return MtgOpusDataset(
        track_file=Path(os.environ["MTG_DATASET_PATH"]) / f"{split}.tsv",
        opus_dir=Path(os.environ["MTG_DATASET_PATH"]) / "opus",
        genres=config.data.genres,
        sampling_rate=config.data.sampling_rate,
        duration=config.data.duration,
        transform=nn.Sequential(
            torchaudio.transforms.Spectrogram(
                n_fft=config.data.n_fft,
                win_length=config.data.n_fft,
                hop_length=config.data.hop_length,
                power=None,
            ),
            ComplexToFeatures(
                ref=config.data.transforms.ref,
                amin=config.data.transforms.amin,
                top_db=config.data.transforms.top_db,
                sigmoid_temp=config.data.transforms.sigmoid_temp,
            ),
        ))


if __name__ == '__main__':
    ds = MtgOpusDataset(track_file=Path(os.environ["MTG_DATASET_PATH"]) / "train.tsv",
                        opus_dir=Path(os.environ["MTG_DATASET_PATH"]) / "opus",
                        genres=None,
                        sampling_rate=22050,
                        duration=10.0,
                        transform=None
                        )

    mtg = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    print("# batches", len(mtg))
    for i, batch in enumerate(mtg):
        print(batch["track"].shape)
        torchaudio.save("sample.mp3", torch.unsqueeze(batch["track"][0], 0), sample_rate=22050)
        break
