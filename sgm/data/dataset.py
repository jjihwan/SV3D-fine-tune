from typing import Optional

import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
import decord
import os
import random
import numpy as np
import torchvision
from torchvision.transforms import ToTensor
from glob import glob
from PIL import Image

try:
    from sdata import create_dataset, create_dummy_dataset, create_loader
except ImportError as e:
    print("#" * 100)
    print("Datasets not yet available")
    print("to enable, we need to add stable-datasets as a submodule")
    print("please use ``git submodule update --init --recursive``")
    print("and do ``pip install -e stable-datasets/`` from the root of this repo")
    print("#" * 100)
    exit(1)


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_datapipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_datapipeline, **self.test_config.loader)

class VideoDataset(Dataset):
    def __init__(self, num_samples=None, width=576, height=576, sample_frames=21, data_root='dataset'):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.data_root = data_root
        self.num_samples = num_samples if num_samples else len(os.listdir(data_root))
        print(f"num_samples: {self.num_samples}")
        self.v_decoder = decord.VideoReader
        
        elevations_deg = [10.0] * sample_frames
        azimuths_deg = np.linspace(0, 360, sample_frames+1)[1:] % 360
        self.polars_rad = [np.deg2rad(90-e) for e in elevations_deg]
        self.azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
        self.azimuths_rad[:-1].sort()

        self.polars_rad = torch.tensor(self.polars_rad)
        self.azimuths_rad = torch.tensor(self.azimuths_rad)



    def __len__(self):
        return self.num_samples

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        start_frame_ind = random.randint(0, total_frames - self.sample_frames)
        end_frame_ind = min(start_frame_ind + self.sample_frames, total_frames)
        
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        # imageio.imwrite('test.png', video_data[0].squeeze(0).permute(1, 2, 0).numpy())
        return video_data
    

    def rand_log_normal(self, shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
        """Draws samples from an lognormal distribution."""
        u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
        return torch.distributions.Normal(loc, scale).icdf(u).exp()
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        data_dirs = sorted(glob(self.data_root + "/*"))

        if len(data_dirs) == 0:
            raise ValueError(
                f"--dataset_path '{self.video_root}' does not contain any .mp4 files.")

        mp4_file = os.path.join(data_dirs[idx], "orbit_frame.mp4")
        last_frame_path = os.path.join(data_dirs[idx], "orbit_frame_0020.png")
        latent_file = mp4_file.replace(".mp4", ".pt")

        video_latent = torch.load(latent_file)

        # video = self.decord_read(mp4_file) # 0~255
        # video = torchvision.transforms.functional.resize(video, (self.height, self.width))
        # #normalize the video to range [-1,1]
        # last_frame = video[-1] / 127.5 - 1 # (1, C, H, W)

        rgba = Image.open(last_frame_path)
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1.0 - rgba_arr[...,-1:])
        last_frame = Image.fromarray((rgb * 255).astype(np.uint8))

        last_frame = ToTensor()(last_frame)
        last_frame = last_frame * 2.0 - 1.0

        cond_frames_without_noise = last_frame
        # cond_sigmas = self.rand_log_normal(shape=[1,], loc=-3.0, scale=0.5)
        cond_sigmas = torch.Tensor([1e-5]) ## ToDo
        cond_frames = torch.rand_like(cond_frames_without_noise) * cond_sigmas + cond_frames_without_noise

        image_only_indicator = torch.zeros(self.sample_frames)
        num_video_frames = self.sample_frames
        cond_sigmas = cond_sigmas.repeat(self.sample_frames)
        
        output_dict= {'video_latent': video_latent, # latent
                'cond_frames_without_noise': cond_frames_without_noise, # image
                'cond_frames': cond_frames, # image
                'cond_aug': cond_sigmas, ## constant?
                'polars_rad': self.polars_rad,
                'azimuths_rad': self.azimuths_rad,
                'image_only_indicator': image_only_indicator,
                'num_video_frames': num_video_frames
                }
        # print()
        # print("[dataloader]")
        # for k, v in output_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        return output_dict
    

class SV3DDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.dataset = VideoDataset(data_root=self.data_root)
        # self.train_data, self.val_data = random_split(self.dataset, (int(1*len(self.dataset)), int(0*len(self.dataset))), generator=torch.Generator().manual_seed(0))
        self.train_data = self.dataset
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=RandomSampler(self.train_data), num_workers=self.num_workers)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=RandomSampler(self.train_data), num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=RandomSampler(self.train_data), num_workers=self.num_workers)
