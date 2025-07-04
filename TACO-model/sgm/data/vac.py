import os  # noqa
import sys  # noqa

# Library imports.
import lovely_tensors
import multiprocessing as mp
import numpy as np
import pytorch_lightning as pl
import time
import random
import json
import math
from PIL import Image
import torch
import torch.nn
import torch.nn.functional
import torch.utils.data
import torchvision
from torch.utils.data.distributed import DistributedSampler
import traceback
from torchvision import transforms
from einops import rearrange
from lovely_numpy import lo
from rich import print

# mp.set_start_method('spawn', force=True)

# import multiprocess
# multiprocess.set_start_method('spawn', force=True)
# torch.utils.data.dataloader.python_multiprocessing = multiprocess
# new_multiprocess_ctx = multiprocess.get_context()

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)


class VACDataset(torch.utils.data.Dataset):

    def __init__(
            self, dset_root, train, force_shuffle=False,
            model_frames=14, input_frames=7, 
            output_frames=14,
            center_crop=True, frame_width=384, frame_height=384,
            input_mode='arbitrary', output_mode='arbitrary',
            motion_bucket_range=[127, 127],
            cond_aug=0.02,
            reverse_prob=0.2, data_gpu=0,
            **kwargs):
        super().__init__()
        # breakpoint()
        self.train = train
        self.dset_root = dset_root
        self.force_shuffle = force_shuffle
        self.model_frames = model_frames
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.center_crop = center_crop
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.motion_bucket_range = motion_bucket_range
        self.cond_aug = cond_aug
        self.reverse_prob = reverse_prob
        self.data_gpu = data_gpu
        self.train_path = os.path.join(self.dset_root, kwargs['train_path'])
        self.val_path = os.path.join(self.dset_root, kwargs['val_path'])
        with open(self.train_path, 'r') as f:
            self.train_list = json.load(f)
        with open(self.val_path, 'r') as f:
            self.val_list = json.load(f)
        # self.data_path = os.path.join(self.dset_root, 'tmp_path.json')
        # with open(self.data_path, 'r') as f:
        #     self.video_list = json.load(f)
        # random.shuffle(self.video_list)
        # tot_videos = len(self.video_list)
        if self.train == 'train':
            self.video_list = self.train_list
            # self.video_list = self.video_list[:math.floor(tot_videos / 100. * 99.)]
            # self.video_list = self.video_list[:10]
            # if not os.path.exists(os.path.join(self.dset_root, 'train_path.json')):
            #     with open(os.path.join(self.dset_root, 'train_path.json'), 'w') as f:
            #         json.dump(self.video_list, f, indent=4)
        elif self.train == 'val':
            self.video_list = self.val_list
            # self.video_list = self.video_list[math.floor(tot_videos / 100. * 99.):]
            # if not os.path.exists(os.path.join(self.dset_root, 'val_path.json')):
            #     with open(os.path.join(self.dset_root, 'val_path.json'), 'w') as f:
            #         json.dump(self.video_list, f, indent=4)
            # print(self.video_list)

        self.dataset_size = len(self.video_list)
        self.avail_fps = 24
        image_transforms = []
        image_transforms.extend([transforms.Resize((self.frame_height, self.frame_width)),
                                transforms.ToTensor(),
                                transforms.Lambda(self.normalize)])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.next_example = None
        # self.total_counter = mp.Value('i', 0)
        self.max_retries = 100
        self.reproject_rgbd = False
    def normalize(self, x):
        return x * 2. - 1.
    def set_next_example(self, *args):
        '''
        For evaluation purposes.
        '''
        # Typically, args = [scene_idx, frame_skip, frame_start, reverse,
        # azimuth_start, azimuth_end, elevation_start, elevation_end, radius_start, radius_end].
        self.next_example = [*args]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # print(len(self.video_list))
        video_path = os.path.join(self.dset_root, self.video_list[idx])
        frame_num = len(os.listdir(os.path.join(video_path, 'occluded_frame')))
        max_frame_start = frame_num - self.model_frames
        frame_start = np.random.randint(0, max_frame_start + 1)
        clip_frames = np.arange(self.model_frames) + frame_start

        reverse = (np.random.rand() < self.reverse_prob)

        if reverse:
            clip_frames = clip_frames[::-1].copy()
        
        input = []
        target = []
        visible_mask = []
        for frame in clip_frames:
            input.append(self.image_transforms(Image.open(os.path.join(video_path, 'occluded_frame', f'occluded_frame_{frame:03d}.png'))))
            target.append(self.image_transforms(Image.open(os.path.join(video_path, 'amodal_object', f'amodal_object_{frame:03d}.png'))))
            visible_mask.append(self.image_transforms(Image.open(os.path.join(video_path, 'visible_mask', f'visible_mask_{frame:03d}.png')).convert('RGB')))
        fps = 6
        motion_amount = 127
        # Now construct the final tensors that will be actually used in the model pipeline.
        data_dict = self.construct_dict(
            input, target, visible_mask, fps, motion_amount)



        # Add extra info / metadata for debugging / logging.
        data_dict['dset'] = torch.tensor([1])
        data_dict['idx'] = torch.tensor([idx])
        data_dict['frame_start'] = torch.tensor([frame_start])
        data_dict['clip_frames'] = torch.tensor(clip_frames)
        data_dict['video_id'] = self.video_list[idx]

        return data_dict

    def construct_dict(self, input, target, visible_mask, fps, motion_amount):
        cond_aug = torch.ones((self.model_frames,), dtype=torch.float32) * self.cond_aug
        # tensor of float32 = all 0.02. dim=14

        # Assign appropriate motion value if model is being trained with synchronized values.
        motion_value = motion_amount
        motion_bucket_id = torch.ones((self.model_frames,), dtype=torch.int32) * motion_value
        # tensor of int32 = all 127. dim=14

        fps_id = torch.ones((self.model_frames,), dtype=torch.int32) * fps
        # fps=6, dim=14
        image_only_indicator = torch.zeros((1, self.model_frames), dtype=torch.float32)
        # tensor of float32 = all 0. dim=(1, 14)

        data_dict = dict()
        data_dict['cond_aug'] = cond_aug.type(torch.float32)
        data_dict['motion_bucket_id'] = motion_bucket_id.type(torch.int32)
        data_dict['fps_id'] = fps_id.type(torch.int32)
        data_dict['image_only_indicator'] = image_only_indicator.type(torch.float32)

        rgb_input = torch.stack(input, dim=0)
        rgb_output = torch.stack(target, dim=0)
        visible_masks = torch.stack(visible_mask, dim=0)
        first_frame = visible_mask[0].repeat(self.model_frames, 1, 1, 1)

        tmp_noise = torch.randn_like(rgb_input)
        cond_frames_with_noise = rgb_input + self.cond_aug * tmp_noise

        data_dict['jpg'] = rgb_output.type(torch.float32)
        data_dict['cond_frames'] = cond_frames_with_noise.type(torch.float32)
        data_dict['cond_frames_without_noise'] = rgb_input.type(torch.float32)
        data_dict['visible_masks'] = visible_masks.type(torch.float32)
        data_dict['first_frame'] = first_frame.type(torch.float32)
        # print('rgb_output:', rgb_output.shape)
        # print('cond_frames_with_noise:', cond_frames_with_noise.shape)
        # print('visible_masks:', visible_masks.shape)
        return data_dict


def collate_fn(example_list):
    collated = torch.utils.data.default_collate(example_list)
    # Correct result by merging batch & temporal dimensions.
    batch = {k: rearrange(v, 'b t ... -> (b t) ...') for (k, v) in collated.items() if k != 'video_id'}
    batch['num_video_frames'] = batch['image_only_indicator'].shape[-1]
    batch['video_id'] = collated['video_id']
    return batch


class VACSynthViewModule(pl.LightningDataModule):

    def __init__(
            self, dset_root, batch_size, num_workers, shuffle=True, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        # print(num_workers)
        self.shuffle = shuffle
        self.train_dataset = VACDataset(
            dset_root, 'train', **kwargs)
        self.val_dataset = VACDataset(
            dset_root, 'val', **kwargs)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        # sampler = DistributedSampler(self.train_dataset)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        # sampler = DistributedSampler(self.train_dataset)
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
