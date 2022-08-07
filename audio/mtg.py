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
    df = df[df.chunk_nr == 0]
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
                 transform=None,
                 target_transform=None
                 ):
        self.track_file = track_file
        self.opus_dir = opus_dir
        self.genres = genres
        self.sampling_rate = sampling_rate
        self.duration = duration
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
                                                  num_frames=int(self.duration * 48000))
        track_orig = torch.mean(track_orig, dim=0)  # stereo to mono
        track = T.resample(track_orig, orig_freq=sample_rate, new_freq=self.sampling_rate)

        # TODO: remove silence

        genres = self.tracks.iloc[idx, 1]
        label = genres[0] if len(genres) > 0 else "nothing"
        if self.transform:
            track = self.transform(track)
        if self.target_transform:
            label = self.target_transform(label)
        return {
            "track": track,
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
        x = torch.stack([x, torch.zeros(x.shape[0], 1, x.shape[2])], dim=1)
        re = rearrange(self.normalizers["real"].inverse_transform(rearrange(x.real, "f t -> t f")), "t f -> f t")
        im = rearrange(self.normalizers["imag"].inverse_transform(rearrange(x.imag, "f t -> t f")), "t f -> f t")
        return re + 1j * im


class InverseSamples(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformation = nn.Sequential(
            InverseNormalizeRealImag(load_normalizers()),
            InverseSpectrogram(n_fft=config.data.n_fft,
                               win_length=config.data.n_fft,
                               hop_length=config.data.hop_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformation(x)


def get_mtg_dataset(split, config):
    normalizers = load_normalizers(Path(__file__).parent / "audio_normalizers.pickel")
    return MtgOpusDataset(
        track_file=Path(os.environ["MTG_DATASET_PATH"]) / f"{split}.tsv",
        opus_dir=os.environ["MTG_DATASET_PATH"] / "opus",
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
            NormalizeRealImag(normalizers=normalizers)
        ))


if __name__ == '__main__':
    ds = MtgOpusDataset(track_file=Path(os.environ["MTG_DATASET_PATH"]) / "train.tsv",
                        opus_dir=Path(os.environ["MTG_DATASET_PATH"]) / "opus",
                        genres=None,
                        sampling_rate=22050,
                        duration=10.0,
                        transform=nn.Sequential(
                            torchaudio.transforms.Spectrogram(
                                n_fft=512,
                                win_length=512,
                                hop_length=256,
                                power=None,
                            ),
                            NormalizeRealImag(
                                normalizers=load_normalizers(Path(__file__).parent / "audio_normalizers.pickel"))
                        )
                        )

    mtg = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    print("# batches", len(mtg))
    for i, batch in enumerate(mtg):
        print(batch["track"].shape)
        print(batch["genre"])
        if i > 5:
            break
