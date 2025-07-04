import os  # noqa
import sys  # noqa
sys.path.insert(0, os.getcwd())  # noqa

# Library imports.
import argparse
import copy
import cv2
import torch
import glob
import joblib
import lovely_tensors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pathlib
import random
import skimage
import skimage.metrics
import sys
import time
from tqdm import tqdm
import traceback
import warnings
from einops import rearrange
from lovely_numpy import lo
from rich import print
import json
from tqdm import TqdmExperimentalWarning
import math
from torchvision import transforms
import torchvision
from PIL import Image
from sgm.util import exists, instantiate_from_config, isheatmap
from torch.utils.data import DataLoader

# Internal imports.
from scripts import eval_utils

lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)


class VACDataset(torch.utils.data.Dataset):

    def __init__(
            self, dset_root, train, force_shuffle=False,
            model_frames=14, input_frames=7, 
            output_frames=14,
            center_crop=True, frame_width=384, frame_height=384,
            input_mode='arbitrary', output_mode='arbitrary',
            motion_bucket_range=[127, 127],
            cond_aug=0.02,
            reverse_prob=0.0, data_gpu=0,
            **kwargs):
        super().__init__()
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
        self.data_path = os.path.join(self.dset_root, 'val.json')
        with open(self.data_path, 'r') as f:
            self.video_list = json.load(f)
        tot_videos = len(self.video_list)
        if self.train == 'train':
            self.video_list = self.video_list[:math.floor(tot_videos / 100. * 99.)]
            # self.video_list = self.video_list[:10]
        elif self.train == 'val':
            print(self.video_list)

        self.dataset_size = len(self.video_list)
        self.avail_fps = 24
        image_transforms = []
        image_transforms.extend([transforms.Resize([self.frame_height, self.frame_width]),
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
        try:
            print(len(self.video_list))
            video_path = os.path.join(self.dset_root, self.video_list[idx])
            frame_num = len(os.listdir(os.path.join(video_path, 'occluded_frame')))
            assert len(os.listdir(os.path.join(video_path, 'occluded_frame'))) == len(os.listdir(os.path.join(video_path, 'visible_mask')))
            max_frame_start = frame_num - self.model_frames
            frame_start = 0
            clip_frames = np.arange(self.model_frames) + frame_start

            reverse = (np.random.rand() < self.reverse_prob)

            if reverse:
                clip_frames = clip_frames[::-1].copy()
            
            input = []
            target = []
            visible_mask = []
            for frame in clip_frames:
                input.append(self.image_transforms(Image.open(os.path.join(video_path, 'occluded_frame', f'occluded_frame_{frame:03d}.png')).convert('RGB')))
                # target.append(self.image_transforms(Image.open(os.path.join(video_path, 'amodal_rgb', f'{frame:02d}.png')).convert('RGB')))
                visible_mask.append(self.image_transforms(Image.open(os.path.join(video_path, 'visible_mask', f'visible_mask_{frame:03d}.png')).convert('RGB')))
            fps = 6
            motion_amount = 127
            # Now construct the final tensors that will be actually used in the model pipeline.
            data_dict = self.construct_dict(
                input, visible_mask, fps, motion_amount)



            # Add extra info / metadata for debugging / logging.
            data_dict['dset'] = torch.tensor([1])
            data_dict['idx'] = torch.tensor([idx])
            data_dict['frame_start'] = torch.tensor([frame_start])
            data_dict['clip_frames'] = torch.tensor(clip_frames)
            data_dict['video_id'] = self.video_list[idx]

            return data_dict
        except:
            idx = 308
            print(len(self.video_list))
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


    def construct_dict(self, input, visible_mask, fps, motion_amount):
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
        rgb_output = torch.zeros_like(rgb_input)
        visible_masks = torch.stack(visible_mask, dim=0)

        tmp_noise = torch.randn_like(rgb_input)
        cond_frames_with_noise = rgb_input + self.cond_aug * tmp_noise

        data_dict['jpg'] = rgb_output.type(torch.float32)
        data_dict['cond_frames'] = cond_frames_with_noise.type(torch.float32)
        data_dict['cond_frames_without_noise'] = rgb_input.type(torch.float32)
        data_dict['visible_masks'] = visible_masks.type(torch.float32)

        return data_dict

def collate_fn(example_list):
    collated = torch.utils.data.default_collate(example_list)
    # Correct result by merging batch & temporal dimensions.
    batch = {k: rearrange(v, 'b t ... -> (b t) ...') for (k, v) in collated.items() if k != 'video_id'}
    batch['num_video_frames'] = batch['image_only_indicator'].shape[-1]
    batch['video_id'] = collated['video_id']
    return batch

def test_args():

    parser = argparse.ArgumentParser()

    # Resource options.
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--gpus', type=str, default='0,1,2,3')
    parser.add_argument('--debug', type=int, default=0)

    # General data options.
    parser.add_argument('--input', type=str, nargs='+',
                        default=[r'../eval/list/cool_videos.txt'],
                        help='One or more paths to video files, and/or directories with images, '
                        'and/or root of evaluation sets, and/or text files with list of examples.')
    parser.add_argument('--output', type=str,
                        default=r'../eval/output/dbg1')

    # General model options.
    parser.add_argument('--config_path', type=str,
                        default=r'configs/infer_kubric.yaml')
    parser.add_argument('--model_path', type=str, nargs='+',
                        default=[r'../pretrained/kubric_gradual_max90.ckpt'],
                        help='One or more paths to trained model weights.')
    parser.add_argument('--use_ema', type=int, default=0)
    parser.add_argument('--autocast', type=int, default=1)

    # Model inference options.
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=14)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--guider_max_scale', type=float, default=1.5)
    parser.add_argument('--guider_min_scale', type=float, default=1.0)
    parser.add_argument('--motion_id', type=int, default=127)
    # ^ NOTE: If motion_bucket_id is synchronized with camera angles during training, this code
    # will take care of automatically setting it (thus overriding the provided value).
    parser.add_argument('--force_custom_mbid', type=int, default=0)
    parser.add_argument('--cond_aug', type=float, default=0.02)
    parser.add_argument('--decoding_t', type=int, default=14)

    # Camera control & frame bounds options.
    parser.add_argument('--delta_azimuth', type=float, default=30.0)
    parser.add_argument('--delta_elevation', type=float, default=15.0)
    parser.add_argument('--delta_radius', type=float, default=0.0)
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_stride', type=int, default=2)
    parser.add_argument('--frame_rate', type=int, default=12)

    # Data processing options.
    parser.add_argument('--frame_width', type=int, default=384)
    parser.add_argument('--frame_height', type=int, default=384)
    parser.add_argument('--center_crop', type=int, default=1)
    parser.add_argument('--save_images', type=int, default=1)
    parser.add_argument('--save_mp4', type=int, default=1)
    # ^ NOTE: Galleries always have MP4 regardless of the save_mp4 setting.
    parser.add_argument('--save_input', type=int, default=1)
    parser.add_argument('--save_uncertainty', type=int, default=1)

    args = parser.parse_args()

    args.gpus = [int(x.strip()) for x in args.gpus.split(',')]

    return args


def load_input(args, worker_idx, example, model_bundle):
    '''
    NOTE: This method supports both known datasets as well as random input videos or images.
    :return input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [-1, 1].
    :return controls (dict).
    :return batch (dict).
    '''
    [model, train_config, test_config, device, model_name] = model_bundle[0:5]

    assert args.frame_start >= 0 and args.frame_stride >= 0 and args.frame_rate >= 0, \
        f'{args.frame_start} {args.frame_stride} {args.frame_rate}'

    controls = np.array([args.frame_start, args.frame_stride, args.frame_rate,
                        args.delta_azimuth, args.delta_elevation, args.delta_radius],
                        dtype=np.float32)
    # NOTE: ^ All camera angle values are in degrees or meters.

    # Pick all 14 frames with contemporaneous input and output.
    Tc = args.num_frames
    clip_frames = np.arange(Tc) * int(controls[1]) + int(controls[0])
    print(f'[gray]{worker_idx}: ')
    print(f'[gray]{worker_idx}: Tc: {Tc} clip_frames: {clip_frames}')
    assert np.all(clip_frames >= 0)

    # NOTE: If this is actually an image, it will be repeated as a still across all frames.
    input_rgb = eval_utils.load_image_or_video(
        example, clip_frames, args.center_crop, args.frame_width, args.frame_height, True)
    input_rgb = (input_rgb + 1.0) / 2.0
    # (Tc, 3, Hp, Wp) array of float32 in [0, 1].

    # motion_bucket = int(train_config.data.params.motion_bucket_range[0])
    # cond_aug = float(train_config.data.params.cond_aug)
    batch = eval_utils.construct_batch(
        input_rgb, controls[3], controls[4], controls[5],
        Tc, controls[2], args.motion_id, args.cond_aug,
        args.force_custom_mbid, model_bundle, device)

    (_, _, Hp, Wp) = input_rgb.shape
    assert Hp % 64 == 0 and Wp % 64 == 0, \
        f'Input resolution must be a multiple of 64, but got {Hp} x {Wp}'

    return (input_rgb, controls, batch)


def run_inference(args, device, model, batch):
    import torch

    autocast_kwargs = eval_utils.prepare_model_inference_params(
        model, device, args.num_steps, args.num_frames,
        args.guider_max_scale, args.guider_min_scale, args.autocast, args.decoding_t)

    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):
            pred_samples = []

            for sample_idx in range(args.num_samples):
                # Perform denoising loop.
                # NOTE: use_ema is False because we already entered the EMA scope before
                # (i.e. when calling process_example which calls run_inference).
                video_dict = model.sample_video(
                    batch, enter_ema=False, limit_batch=False)

                output_dict = dict()
                output_dict['cond_rgb'] = video_dict['cond_video'].detach().cpu().numpy()
                # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                output_dict['sampled_rgb'] = video_dict['sampled_video'].detach().cpu().numpy()
                # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                output_dict['sampled_latent'] = video_dict['sampled_z'].detach().cpu().numpy()
                # (Tcm, 4, Hl, Wl) = (14, 4, 32, 48) array of float32.

                pred_samples.append(output_dict)

    return pred_samples


def calculate_metrics(args, pred_samples):
    '''
    :param pred_samples: list of dicts with keys:
        cond_rgb, sampled_rgb, sampled_latent.
    '''
    # NOTE: This subroutine is a bit rudimentary, because it does not include baseline metrics;
    # see more advanced scripts for that.
    S = len(pred_samples)

    if S >= 1:
        pred_samples_rgb = np.stack([x['sampled_rgb'] for x in pred_samples], axis=0)
        # (S, Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    else:
        pred_samples_rgb = []

    uncertainty = np.mean(np.std(pred_samples_rgb, axis=0), axis=1)
    # (Tcm, Hp, Wp) array of float32 in [0, 1].
    frame_diversity = np.mean(uncertainty, axis=(1, 2))
    # (Tcm) array of float32 in [0, 1].
    mean_diversity = np.mean(frame_diversity)
    # single float.

    metrics_dict = dict()
    metrics_dict['frame_diversity'] = frame_diversity
    metrics_dict['mean_diversity'] = mean_diversity

    return (metrics_dict, uncertainty)


def get_controls_friendly(controls):
    frame_start = int(controls[0])
    frame_stride = int(controls[1])
    frame_rate = int(controls[2])
    delta_azimuth = float(controls[3])
    delta_elevation = float(controls[4])
    delta_radius = float(controls[5])
    # NOTE: ^ All values are in degrees or meters.

    if delta_azimuth != 0.0 or delta_elevation != 0.0 or delta_radius != 0.0:
        nonzero = True
        title = f'A {delta_azimuth:.1f} E {delta_elevation:.1f} R {delta_radius:.1f}'
        filename = (f'fs{frame_start}_fr{frame_rate}_az{delta_azimuth:.1f}'
                    f'_el{delta_elevation:.1f}_rd{delta_radius:.1f}')

    else:
        nonzero = False
        title = f'FPS {frame_rate}'
        filename = f'fs{frame_start}_fr{frame_rate}'

    return (nonzero, title, filename)


def create_visualizations(
        args, input_rgb, controls_friendly, pred_samples,
        metrics_dict, uncertainty, model_name):
    '''
    :param input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param pred_samples: List of dict.
    :param uncertainty: (Tcm, Hp, Wp) array of float32 in [0, 1].
    '''
    (Tcm, _, Hp, Wp) = input_rgb.shape
    S = len(pred_samples)

    if controls_friendly[0]:
        target_title = f'Target ({controls_friendly[1]})'
    else:
        target_title = f'Target'

    input_rgb = rearrange(input_rgb, 't c h w -> t h w c')
    # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].

    pred_samples_rgb = []
    uncertainty_rgb = None

    if S >= 1:
        frame_diversity = metrics_dict['frame_diversity']

        pred_samples_rgb = [
            rearrange(x['sampled_rgb'], 't c h w -> t h w c') for x in pred_samples]
        # (S, Tcm, Hp, Wp, 3) array of float32 in [0, 1].
        pred_samples_latent = [
            rearrange(x['sampled_latent'], 't c h w -> t h w c') for x in pred_samples]
        # (S, Tcm, Hl, Wl, 4) array of float32.

        if uncertainty is not None:
            used_uncertainty = np.clip(uncertainty * 3.0, 0.0, 1.0)
            # (Tcm, Hp, Wp) array of float32 in [0, 1].
            uncertainty_rgb = plt.cm.magma(used_uncertainty)[..., 0:3]
            # (Tcm, Hp, Wp, 3) array of float32 in [0, 1].

        # NOTE: The PCA visualization is computed on all samples together to allow for comparison.
        pred_samples_latent_pca = eval_utils.quick_pca(
            np.stack(pred_samples_latent, axis=0), k=3, normalize=[0.0, 1.0])
        # (S, Tcm, Hl, Wl, 3) array of float32 in [0, 1].

        (_, Hl, Wl, _) = pred_samples_latent[0].shape
        F = Hp // Hl
        pred_samples_latent_vis = np.repeat(pred_samples_latent_pca, F, axis=2)
        pred_samples_latent_vis = np.repeat(pred_samples_latent_vis, F, axis=3)
        assert pred_samples_latent_vis.shape == (S, Tcm, Hp, Wp, 3)

    # NOTE: I disabled some of these for GCD, 2024 code publication since it might be a bit excessive for
    # most use cases, but feel free to re-enable them and/or implement your own stuff.

    rich1_frames = []
    rich2_frames = []
    rich4_frames = []
    rich5_frames = []
    font_size = 1.0

    for t in range(Tcm):
        # Rich 1: Input, Blank || Output 1, Output 2 || Output 3, Output 4.
        if S <= 2:
            canvas1 = np.zeros((Hp * 2 + 80, Wp * 2, 3), dtype=np.float32)
        else:
            canvas1 = np.zeros((Hp * 2 + 80, Wp * 3, 3), dtype=np.float32)

        eval_utils.draw_text(canvas1, (20, 5), (0.5, 0.0),
                             f'Input (Frame {t})', (1, 1, 1), font_size)
        eval_utils.draw_text(canvas1, (Hp + 60, 5), (0.5, 0.0),
                             target_title, (1, 1, 1), font_size)

        canvas1[40:Hp + 40, 0:Wp] = input_rgb[t]

        if S >= 1:
            eval_utils.draw_text(
                canvas1, (20, Wp + 5), (0.5, 0.0),
                f'Output 1',
                (1, 1, 1), font_size)
            canvas1[40:Hp + 40, Wp:Wp * 2] = pred_samples_rgb[0][t].copy()

        if S >= 2:
            eval_utils.draw_text(
                canvas1, (Hp + 60, Wp + 5), (0.5, 0.0),
                f'Output 2',
                (1, 1, 1), font_size)
            canvas1[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] = pred_samples_rgb[1][t].copy()

        if S >= 3:
            eval_utils.draw_text(
                canvas1, (20, Wp * 2 + 5), (0.5, 0.0),
                f'Output 3',
                (1, 1, 1), font_size)
            canvas1[40:Hp + 40, Wp * 2:Wp * 3] = pred_samples_rgb[2][t].copy()

        if S >= 4:
            eval_utils.draw_text(
                canvas1, (Hp + 60, Wp * 2 + 5), (0.5, 0.0),
                f'Output 4',
                (1, 1, 1), font_size)
            canvas1[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] = pred_samples_rgb[3][t].copy()

        rich1_frames.append(canvas1)

        # Rich 2: Input || Output 1.
        if S >= 1:
            canvas2 = canvas1[0:Hp + 40, 0:Wp * 2].copy()
            canvas2[0:40, Wp:Wp * 2] = 0.0
            eval_utils.draw_text(canvas2, (20, Wp + 5), (0.5, 0.0),
                                 f'Output ({model_name})', (1, 1, 1), font_size)

            rich2_frames.append(canvas2)

        # Rich 4: Input, Blank || Output 1, Output 2 || Latent 1, Latent 2.
        # if S >= 1:
        #     canvas4 = np.zeros((Hp * 2 + 80, Wp * 3, 3), dtype=np.float32)
        #     canvas4[:, 0:Wp * 2] = canvas1[:, 0:Wp * 2].copy()

        #     eval_utils.draw_text(canvas4, (20, Wp * 2 + 5), (0.5, 0.0),
        #                          f'Latent 1', (1, 1, 1), font_size)
        #     canvas4[40:Hp + 40, Wp * 2:Wp * 3] = pred_samples_latent_vis[0][t].copy()

        #     if S >= 2:
        #         eval_utils.draw_text(canvas4, (Hp + 60, Wp * 2 + 5), (0.5, 0.0),
        #                              f'Latent 2', (1, 1, 1), font_size)
        #         canvas4[Hp + 80:Hp * 2 + 80, Wp * 2:Wp * 3] = pred_samples_latent_vis[1][t].copy()

        #     rich4_frames.append(canvas4)

        # Rich 5: Input, Blank ||  Delta, Uncert.
        if S >= 2 and uncertainty_rgb is not None:
            delta_rgb = np.abs(pred_samples_rgb[0][t] - pred_samples_rgb[1][t]) * 2.0
            canvas5 = np.zeros((Hp * 2 + 80, Wp * 2, 3), dtype=np.float32)
            canvas5[:, 0:Wp] = canvas1[:, 0:Wp].copy()

            eval_utils.draw_text(canvas5, (20, Wp + 5), (0.5, 0.0),
                                 f'Delta (Div {frame_diversity[t]:.3f})', (1, 1, 1), font_size)
            canvas5[40:Hp + 40, Wp:Wp * 2] = pred_samples_rgb[0][t] * 0.3  # Darken output.
            canvas5[40:Hp + 40, Wp:Wp * 2] += delta_rgb * 0.8

            eval_utils.draw_text(canvas5, (Hp + 60, Wp + 5), (0.5, 0.0),
                                 f'Uncertainty (Div {frame_diversity[t]:.3f})', (1, 1, 1), font_size)
            canvas5[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] = pred_samples_rgb[1][t] * 0.3  # Darken output.
            canvas5[Hp + 80:Hp * 2 + 80, Wp:Wp * 2] += uncertainty_rgb[t] * 0.8

            rich5_frames.append(canvas5)

    # Organize & return results.
    vis_dict = dict()

    # Pause a tiny bit at the beginning and end for less jerky looping.
    rich1_frames = [rich1_frames[0]] + rich1_frames + [rich1_frames[-1]] * 2
    rich1_frames = np.stack(rich1_frames, axis=0)
    rich1_frames = np.clip(rich1_frames, 0.0, 1.0)
    vis_dict['rich1'] = rich1_frames

    if len(rich2_frames) > 0:
        rich2_frames = [rich2_frames[0]] + rich2_frames + [rich2_frames[-1]] * 2
        rich2_frames = np.stack(rich2_frames, axis=0)
        rich2_frames = np.clip(rich2_frames, 0.0, 1.0)
        vis_dict['rich2'] = rich2_frames

    if len(rich4_frames) > 0:
        rich4_frames = [rich4_frames[0]] + rich4_frames + [rich4_frames[-1]] * 2
        rich4_frames = np.stack(rich4_frames, axis=0)
        rich4_frames = np.clip(rich4_frames, 0.0, 1.0)
        vis_dict['rich4'] = rich4_frames

    if len(rich5_frames) > 0:
        rich5_frames = [rich5_frames[0]] + rich5_frames + [rich5_frames[-1]] * 2
        rich5_frames = np.stack(rich5_frames, axis=0)
        rich5_frames = np.clip(rich5_frames, 0.0, 1.0)
        vis_dict['rich5'] = rich5_frames

    vis_dict['input'] = input_rgb
    vis_dict['output'] = pred_samples_rgb
    if uncertainty_rgb is not None:
        vis_dict['uncertainty'] = uncertainty_rgb

    return vis_dict


def save_results(args, metrics_dict, vis_dict, controls, output_fp1, output_fp2):
    vis_fps = (6 + controls[2]) // 2
    eval_utils.write_video_and_frames(
        vis_dict['rich1'], dst_dp=output_fp1 + '_gal', fps=vis_fps,
        save_images=False, save_mp4=True, quality=9)

    if 'rich2' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich2'], dst_dp=output_fp1 + '_io', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich4' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich4'], dst_dp=output_fp1 + '_latent', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if 'rich5' in vis_dict:
        eval_utils.write_video_and_frames(
            vis_dict['rich5'], dst_dp=output_fp1 + '_uncert', fps=vis_fps,
            save_images=False, save_mp4=True, quality=9)

    if args.save_images or args.save_mp4:
        if args.save_input:
            eval_utils.write_video_and_frames(
                vis_dict['input'], dst_dp=output_fp2 + '_input', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        for s in range(args.num_samples):
            eval_utils.write_video_and_frames(
                vis_dict['output'][s], dst_dp=output_fp2 + f'_pred_s{s}', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)

        if args.save_uncertainty and 'uncertainty' in vis_dict:
            eval_utils.write_video_and_frames(
                vis_dict['uncertainty'], dst_dp=output_fp2 + '_uncert', fps=vis_fps,
                save_images=args.save_images, save_mp4=args.save_mp4, quality=9)


def process_example(args, worker_idx, example_idx, example, model_bundle):
    (model, train_config, test_config, device, model_name) = model_bundle[0:5]

    # Load & preprocess input frames.
    print()
    print(f'[yellow]{worker_idx}: Loading input frames from {example}...')
    start_time = time.time()
    (input_rgb, controls, batch) = load_input(
        args, worker_idx, example, model_bundle)
    print(f'[magenta]{worker_idx}: Loading frames took {time.time() - start_time:.2f}s')

    # Run inference.
    print()
    print(f'[cyan]{worker_idx}: Running SVD model on selected video clip...')
    start_time = time.time()
    if args.num_samples >= 1:
        pred_samples = run_inference(
            args, device, model, batch)
    else:
        pred_samples = []
    print(f'[magenta]{worker_idx}: Inference took {time.time() - start_time:.2f}s')

    # Calculate metrics.
    if args.num_samples >= 1:
        print()
        print(f'[cyan]{worker_idx}: Calculating metrics...')
        start_time = time.time()
        (metrics_dict, uncertainty) = calculate_metrics(
            args, pred_samples)
        print(f'[magenta]{worker_idx}: Metrics took {time.time() - start_time:.2f}s')
    else:
        metrics_dict = dict()
        uncertainty = None

    # Create rich inference visualization.
    print()
    print(f'[yellow]{worker_idx}: Creating rich visualizations...')
    start_time = time.time()
    controls_friendly = get_controls_friendly(controls)
    vis_dict = create_visualizations(
        args, input_rgb, controls_friendly, pred_samples,
        metrics_dict, uncertainty, model_name)
    print(f'[magenta]{worker_idx}: Visualizations took {time.time() - start_time:.2f}s')

    # Prepare output directory.
    test_tag = os.path.basename(args.output).split('_')[0]
    output_fn = os.path.splitext(os.path.basename(example))[0]
    output_fn = output_fn.replace('_p0', '')
    output_fn = output_fn.replace('_rgb', '')

    output_fn1 = f'{test_tag}_{example_idx:03d}_n{model_name}'
    output_fn1 += f'_{output_fn}'
    output_fn2 = output_fn1  # Save shorter name for extra data.

    # Contains either just frame rate or frame rate + camera controls.
    output_fn1 += f'_{controls_friendly[2]}'

    # if args.num_samples >= 1:
    #     psnr = np.nanmean(metrics_dict['mean_psnr'])
    #     ssim = np.nanmean(metrics_dict['mean_ssim'])
    #     diversity = metrics_dict['mean_diversity']
    #     # output_fn1 += f'_psnr{psnr:.2f}_ssim{ssim:.3f}_div{diversity:.3f}'

    # For more prominent / visible stuff:
    output_fp1 = os.path.join(args.output, output_fn1)

    # For less prominent but still useful stuff:
    output_fp2 = os.path.join(args.output, 'extra', output_fn2)

    # Save results to disk.
    print()
    print(f'[yellow]{worker_idx}: Saving results to disk...')
    start_time = time.time()
    save_results(args, metrics_dict, vis_dict, controls, output_fp1, output_fp2)
    print(f'[magenta]{worker_idx}: Saving took {time.time() - start_time:.2f}s')

    return True


def worker_fn(args, worker_idx, num_workers, gpu_idx, model_path, example_list):

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    # Only now can we import torch.
    import torch
    from sgm.util import instantiate_from_config
    torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)

    # Update CPU affinity.
    eval_utils.update_os_cpu_affinity(worker_idx, num_workers)

    if not (os.path.exists(model_path)) and '*' in model_path:
        used_model_path = sorted(glob.glob(model_path))[-1]
        print(f'[orange3]{worker_idx}: Warning: Parsed {model_path} '
              f'to assumed latest checkpoint {used_model_path}')
    else:
        used_model_path = model_path

    print()
    print(f'[cyan]{worker_idx}: Loading SVD model from {used_model_path} on GPU {gpu_idx}...')
    start_time = time.time()

    device = args.device
    if 'cuda' in device:
        device = f'cuda:{gpu_idx}'

    # Initialize model.
    model_bundle = eval_utils.load_model_bundle(
        device, args.config_path, used_model_path, args.use_ema,
        num_steps=args.num_steps, num_frames=args.num_frames,
        max_scale=args.guider_max_scale, min_scale=args.guider_min_scale,
        verbose=(worker_idx == 0))
    (model, train_config, test_config, device, model_name) = model_bundle[0:5]

    print(f'[magenta]{worker_idx}: Loading SVD model took {time.time() - start_time:.2f}s')

    eval_utils.warn_resolution_mismatch(train_config, args.frame_width, args.frame_height)

    # Start iterating over all videos.
    if args.debug:
        to_loop = tqdm.tqdm(list(enumerate(example_list)))
    else:
        to_loop = tqdm.rich.tqdm(list(enumerate(example_list)))

    # Enable EMA scope early on to avoid constant shifting around of weights.
    with model.ema_scope('Testing'):

        for (i, example) in to_loop:
            example_idx = i * num_workers + worker_idx

            try:
                process_example(args, worker_idx, example_idx, example, model_bundle)

            except Exception as e:
                print(f'[red]{worker_idx}: Error processing {example}: {e}')
                print(f'[red]Traceback: {traceback.format_exc()}')
                print(f'[red]Skipping...')
                continue

    print()
    print(f'[cyan]{worker_idx}: Done!')
    print()


def main(args):

    # Save the arguments to this training script.
    args_fp = os.path.join(args.output, 'args_infer.json')
    eval_utils.save_json(vars(args), args_fp)
    print(f'[yellow]Saved script args to {args_fp}')

    # Load list of videos to process (not the pixels themselves yet).
    print()
    print(f'[yellow]Parsing list of individual examples from {args.input}...')
    start_time = time.time()

    # examples = eval_utils.get_list_of_input_images_or_videos(args.input)
    # print(f'[yellow]Found {len(examples)} examples '
    #       f'(counting both video files and/or image folders).')

    # print(f'[magenta]Loading data list took {time.time() - start_time:.2f}s')

    # assert len(examples) > 0, f'No examples found in {args.input}!'

    from omegaconf import OmegaConf

    # Load inference config & diffusion model.
    test_config = OmegaConf.load(args.config_path)
    test_config.model.params.conditioner_config.params.emb_models[0].\
        params.open_clip_embedding_config.params.init_device = "cuda"
    test_config.model.params.ckpt_path = args.model_path[0]
    # NOTE: This decides which keys to load and when, so it is important to get right!
    test_config.model.params.use_ema = bool(args.use_ema)
    test_config.model.params.ckpt_has_ema = bool(args.use_ema)

    # Here, we are setting the best known values so far to start off.
    test_config.model.params.sampler_config.params.num_steps = args.num_steps
    test_config.model.params.sampler_config.params.guider_config.params.num_frames = args.num_frames
    test_config.model.params.sampler_config.params.guider_config.params.max_scale = args.guider_max_scale
    test_config.model.params.sampler_config.params.guider_config.params.min_scale = args.guider_min_scale
    test_config.model.params.sampler_config.params.device = "cuda"

    with torch.device("cuda"):
        model = instantiate_from_config(test_config.model).to("cuda").eval()
    data = VACDataset(dset_root=test_config.data.params.dset_root, train='val')
    val_dataloader = DataLoader(dataset=data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # cur_num = 0
    for i, batch in enumerate(tqdm(val_dataloader)):
        # breakpoint()
        tmp = {}
        tmp['video_id'] = batch['video_id'].copy()
        tmp['frame_start'] = int(np.array(batch['frame_start'].detach().cpu()))
        batch = {k: v.to("cuda") for k, v in batch.items() if k != 'video_id' and k != 'num_video_frames'}
        batch['num_video_frames'] = 14
        autocast_kwargs = eval_utils.prepare_model_inference_params(
            model, "cuda", args.num_steps, args.num_frames,
            args.guider_max_scale, args.guider_min_scale, args.autocast, args.decoding_t)
        # breakpoint()
        
        with torch.no_grad():
            with torch.autocast(**autocast_kwargs):
                cur_num = 0
                
                pred_samples = []

                for sample_idx in range(args.num_samples):
                    os.makedirs(os.path.join(args.output, f"{tmp['video_id'][0]}_{cur_num:01d}"), exist_ok=True)
                    with open(os.path.join(args.output, f"{tmp['video_id'][0]}_{cur_num:01d}", 'info.json'), 'w') as f:
                        json.dump(tmp, f, indent=4)
                    # Perform denoising loop.
                    # NOTE: use_ema is False because we already entered the EMA scope before
                    # (i.e. when calling process_example which calls run_inference).
                    video_dict = model.sample_video(
                        batch, enter_ema=False, limit_batch=False)
                    output_dict = dict()
                    output_dict['cond_rgb'] = video_dict['cond_video'].detach().cpu().numpy()
                    # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                    output_dict['sampled_rgb'] = video_dict['sampled_video'].detach().cpu().numpy()
                    # (Tcm, 3, Hp, Wp) = (14, 3, 256, 384) array of float32 in [0, 1].
                    output_dict['sampled_latent'] = video_dict['sampled_z'].detach().cpu().numpy()
                    # (Tcm, 4, Hl, Wl) = (14, 4, 32, 48) array of float32.
                    video_dict['vertcat'] = \
                    torch.cat([video_dict['cond_video'], video_dict['sampled_video']], dim=2)
                    # breakpoint()
                    eval_utils.write_video_and_frames(video_dict['vertcat'].permute(0, 2, 3, 1).detach().cpu().numpy(), 
                                                      dst_dp=os.path.join(args.output, f"{tmp['video_id'][0]}_{cur_num:01d}"), 
                                                      save_amodal=True, save_images=True, save_mp4=False)
                    cur_num += 1

    print(f'[magenta]Everything took {time.time() - start_time:.2f}s')

    print()
    print(f'[cyan]Done!')
    print()


if __name__ == '__main__':

    args = test_args()

    main(args)

    pass
