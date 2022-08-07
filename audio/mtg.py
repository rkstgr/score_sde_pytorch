import os
from typing import Union, List

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as T

from pathlib import Path


def _load_tracks(track_file: Path, genres: Union[str, List[str], None]) -> pd.DataFrame:
    df = pd.read_csv(track_file, sep="\t")
    df = df[df.chunk_nr == 0]
    print("Loaded {} tracks".format(len(df)))
    if genres:
        df.genres = df.genres.apply(lambda x: eval(x))
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
        label = genres[0] if len(genres) == 1 else None
        if self.transform:
            track = self.transform(track)
        if self.target_transform:
            label = self.target_transform(label)
        return {
            "track": track,
            "genre": label
        }


if __name__ == '__main__':
    mtg = MtgOpusDataset(track_file=Path(os.environ["MTG_DATASET_PATH"]) / "train.tsv",
                         opus_dir=Path(os.environ["MTG_DATASET_PATH"]) / "opus",
                         genres=None,
                         sampling_rate=22050,
                         duration=10.0)
    print(len(mtg))
    print(mtg[0]["track"].shape)
