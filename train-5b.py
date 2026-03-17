# !/bin/python3
# isort: skip_file
import argparse
import math
import os
import time
from collections import deque
from copy import deepcopy
import torch.nn.functional as F
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
import gc
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
import bitsandbytes as bnb
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from PIL import Image
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import copy

from hyvideo.diffusion import load_denoiser
from fastvideo.dataset.latent_datasets import (LatentDataset)
from fastvideo.dataset.t2v_datasets import (StableVideoAnimationDataset)

from fastvideo.distill.solver import EulerSolver, extract_into_tensor
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import linear_quadratic_schedule
from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint,
                                        save_lora_checkpoint, resume_checkpoint, resume_training)
from fastvideo.utils.communications import (broadcast,
                                            sp_parallel_dataloader_wrapper)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing,
                                       get_dit_fsdp_kwargs,
                                      get_discriminator_fsdp_kwargs,
                                      get_DINO_fsdp_kwargs)
from fastvideo.utils.load import load_transformer,load_transformer_small
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group,
                                             get_sequence_parallel_state,
                                             initialize_sequence_parallel_state
                                             )
from fastvideo.utils.validation import log_validation
from fastvideo.utils.load import load_text_encoder, load_vae
import time
import torch.distributed as dist

from fastvideo.models.hunyuan.modules.t5 import T5EncoderModel
from fastvideo.models.hunyuan.modules.clip import CLIPModel
from fastvideo.models.hunyuan.modules.model import WanModel
from fastvideo.models.hunyuan.modules.vae import WanVAE

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

from transformers import AutoModel, AutoTokenizer
try:
    import wandb
except ImportError:
    wandb = None


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

def wandb_log(metrics, step=None):
    if dist.get_rank() == 0 and wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def get_norm(model_pred, norms, gradient_accumulation_steps):
    fro_norm = (
        torch.linalg.matrix_norm(model_pred, ord="fro") /  # codespell:ignore
        gradient_accumulation_steps)
    largest_singular_value = (torch.linalg.matrix_norm(model_pred, ord=2) /
                              gradient_accumulation_steps)
    absolute_mean = torch.mean(
        torch.abs(model_pred)) / gradient_accumulation_steps
    absolute_max = torch.max(
        torch.abs(model_pred)) / gradient_accumulation_steps
    dist.all_reduce(fro_norm, op=dist.ReduceOp.AVG)
    dist.all_reduce(largest_singular_value, op=dist.ReduceOp.AVG)
    dist.all_reduce(absolute_mean, op=dist.ReduceOp.AVG)
    norms["fro"] += torch.mean(fro_norm).item()  # codespell:ignore
    norms["largest singular value"] += torch.mean(
        largest_singular_value).item()
    norms["absolute mean"] += absolute_mean.item()
    norms["absolute max"] += absolute_max.item()

def latent_collate_function(latents,prompt_embeds,prompt_attention_masks):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask
    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0,
                max_t - latent.shape[1],
                0,
                max_h - latent.shape[2],
                0,
                max_w - latent.shape[3],
            ),
        ) for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0

    #prompt_embeds = torch.stack(prompt_embeds, dim=0)
    #prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    latents = torch.stack(latents, dim=0)
    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks


def video_collate_function(batch):
    """Collate fn for StableVideoAnimationDataset.
    Each item: (pixel_values, pixel_values_ref_img, caption, keys, mouse, videoid)
    pixel_values shape: [T, C, H, W] — T varies per sample.
    Truncates to the shortest T in the batch so torch.stack succeeds.
    """
    pixel_values, ref_imgs, captions, keys, mouse, videoids = zip(*batch)
    min_t = min(v.shape[0] for v in pixel_values)
    pixel_values = torch.stack([v[:min_t] for v in pixel_values], dim=0)
    ref_imgs = torch.stack(list(ref_imgs), dim=0)
    return pixel_values, ref_imgs, list(captions), list(keys), list(mouse), list(videoids)


import torchvision
    
from diffusers.video_processor import VideoProcessor

import torch
import numpy as np
from diffusers.utils import export_to_video

def scale(vae,latents):
    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
        
    # has_latents_mean = (hasattr(vae.model.config, "latents_mean")
    #                         and vae.model.config.latents_mean is not None)
    # has_latents_std = (hasattr(vae.model.config, "latents_std")
    #                        and vae.model.config.latents_std is not None)
    # if has_latents_mean and has_latents_std:
    #     latents_mean = (torch.tensor(vae.model.config.latents_mean).view(
    #             1, 12, 1, 1, 1).to(latents.device, latents.dtype))
    #     latents_std = (torch.tensor(vae.model.config.latents_std).view(
    #             1, 12, 1, 1, 1).to(latents.device, latents.dtype))
    #     latents = latents * latents_std / vae.model.config.scaling_factor + latents_mean
    # else:
    #     latents = latents / vae.model.config.scaling_factor
    # with torch.autocast("cuda", dtype=vae.dtype):
    with torch.no_grad():
        video = vae.decode([latents.to(torch.float32)])[0]
    video_processor = VideoProcessor(
        vae_scale_factor=vae_spatial_scale_factor)
    #print(video.shape,video)
    video = video_processor.postprocess_video(video.unsqueeze(0), output_type="pil")
    return video

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma

# 转换为 PIL.Image
def tensor_to_pil(tensor):
    # 1. 转换为 NumPy 数组
    array = ((tensor+1)/2.0).detach().cpu().numpy()  # 如果 tensor 在 GPU 上，先移到 CPU
    
    # 2. 调整形状为 (H, W, C)
    array = np.transpose(array, (1, 2, 0))  # 从 (C, H, W) 变为 (H, W, C)
    
    # 3. 转换为 [0, 255] 范围并转为 uint8
    array = (array * 255).astype(np.uint8)
    
    # 4. 创建 PIL 图像
    return Image.fromarray(array)


from packaging import version as pver
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

import random


import pandas as pd

def get_caption(csv_path, video_file):
    """根据videoFile获取caption, 包含完整异常处理"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 检查必需列是否存在
        if 'videoFile' not in df.columns or 'caption' not in df.columns:
            raise ValueError("CSV文件中缺少'videoFile'或'caption'列")

        # 查询匹配项
        matches = df.loc[df['videoFile'] == video_file, 'caption']

        if len(matches) == 0:
            raise ValueError(f"未找到videoFile为'{video_file}'的记录")

        return matches.values[0]
    
    except FileNotFoundError:
        print(f"错误：文件'{csv_path}'不存在")
    except pd.errors.EmptyDataError:
        print("错误：CSV文件为空")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")
    return None

from wan23.utils.utils import best_output_size

def extract_first_frame_from_latents(latents):
    """
    直接从(C,F,H,W)的latents张量提取首帧并转为PIL图像
    :param latents: 输入潜变量张量，形状为(C,F,H,W)
    :return: 首帧PIL图像（自动转换为[0,255]范围）
    """
    # 确保输入是4D张量
    assert len(latents.shape) == 4, "Input must be (C,F,H,W) tensor"
    
    # 提取首帧并反归一化 [-1,1]->[0,255]
    first_frame = latents[:, 0, :, :]  # 取第0帧 -> (C,H,W)
    first_frame = (first_frame + 1) * 127.5  # 数值映射到[0,255]
    
    # 转换为PIL.Image
    first_frame = first_frame.clamp(0, 255).byte()  # 确保数值范围有效
    first_frame = first_frame.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
    return Image.fromarray(first_frame)

def distill_one_step_t2i(
    transformer,
    result_list,
    prompt_all,
    model_type,
    teacher_transformer,
    ema_transformer,
    optimizer,
    discriminator,
    discriminator_optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    solver,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    num_euler_timesteps,
    multiphase,
    not_apply_cfg_solver,
    distill_cfg,
    ema_decay,
    pred_decay_weight,
    pred_decay_type,
    hunyuan_teacher_disable_cfg,
    device,
    vae=None,
    text_encoder=None,
    clip = None,
    source_idx_double=None,
    source_idx_single=None,
    step=None,
    step1=None,
    step2=None,
    wan_i2v=None,
    denoiser=None,
    pipe=None,
    camption_model = None,
    tokenizer = None,
    rank = None,
    world_size = None,
    caption_img_dir = None,
    sample_output_dir = None,
    fps = 16,
    validation_steps = 16,
):
    total_loss = 0.0
    optimizer.zero_grad()
    model_pred_norm = {
        "fro": 0.0, 
        "largest singular value": 0.0,
        "absolute mean": 0.0,
        "absolute max": 0.0,
    }

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
    

    for _ in range(gradient_accumulation_steps):
        
        rand_num = random.random()  # i2v or v2v
      
        rank = dist.get_rank()
        rand_num = torch.ones(1)

        if rank == 0:
            rand_num = random.random()  # i2v or v2v
            s1 = torch.tensor([rand_num])
        else:
            s1 = torch.ones(1, dtype=torch.float)

        s1 = s1.to(device)
        dist.broadcast(s1, src=0)
        rand_num = float(s1)


        rand_numca = random.random()  # i2v or v2v
        (
            pixel_values_vid,
            pixel_values_ref_img,
            caption,
            K_ctrl,
            c2w_ctrl,
            videoid,
        ) = next(loader)
        with torch.no_grad():
            pixel_values_vid = pixel_values_vid.squeeze().permute(1,0,2,3).contiguous().to(device)
            pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
            latents = pixel_values_vid 
            latents = latents[:,-32:]
            latents = F.interpolate(latents, size=(704, 1280), mode='bilinear', align_corners=False)
        frame = ( latents.shape[1] - 1 )*4 + 1
        img1,img2,img3,img4 = extract_first_frame_from_latents(latents[:,0:1]),extract_first_frame_from_latents(latents[:,7:8]),\
        extract_first_frame_from_latents(latents[:,15:16]),extract_first_frame_from_latents(latents[:,31:32])

        latents = wan_i2v.vae.encode([latents.to(device)])[0]

        os.makedirs(caption_img_dir, exist_ok=True)
        path1 = os.path.join(caption_img_dir, f"_{step}_{rank}_1.jpg")
        img1.save(path1)
        path2 = os.path.join(caption_img_dir, f"_{step}_{rank}_2.jpg")
        img2.save(path2)
        path3 = os.path.join(caption_img_dir, f"_{step}_{rank}_3.jpg")
        img3.save(path3)
        path4 = os.path.join(caption_img_dir, f"_{step}_{rank}_4.jpg")
        img4.save(path4)

        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
        pixel_values1 = load_image(path1, max_num=12).to(torch.bfloat16).to(device)
        pixel_values2 = load_image(path2, max_num=12).to(torch.bfloat16).to(device)
        pixel_values3 = load_image(path3, max_num=12).to(torch.bfloat16).to(device)
        pixel_values4 = load_image(path4, max_num=12).to(torch.bfloat16).to(device)
        pixel_values = torch.cat((pixel_values1, pixel_values2, pixel_values3, pixel_values4), dim=0)

        # question = '<image>\nWatch the given egocentric (first-person) video (multi-image) and write a detailed, content-rich caption of approximately 70 words for video generation. Describe the scene, focusing solely on visible people, objects, scenery, weather, lighting, atmosphere, and activities, while avoiding any mention of camera movement, lens changes, or filming techniques.'
        question = '<image>\nWatch the given egocentric (first-person) hand-object interaction sequence (multi-image) and write a detailed, content-rich caption of approximately 70 words for video generation. Describe the scene, focusing on the hand actions, objects being manipulated, surrounding environment, lighting, and atmosphere, while avoiding any mention of camera movement, lens changes, or filming techniques.'

        response, history = camption_model.chat(tokenizer, pixel_values, question, generation_config,
                                           history=None, return_history=True)
 
        caption = "realistic style. " + caption[0] + response
        # video_id = "city_walk_"
        video_id = "egocentric_"

     
        with torch.no_grad():
            # Jump to ./wan/image2video.py
            arg_c, arg_null, noise = wan_i2v.generate(
                       caption,
                       frame_num=frame )
        model_input = latents

        # MVDT
        # xt, t, model_output, loss_dict_mask, x0, t  = denoiser.training_losses(
        #                 transformer,
        #                 latents,
        #                 arg_c,
        #                 n_tokens=None,
        #                 i2v_mode=None,
        #                 cond_latents=None,
        #                 args=args,
        #                 training_cache=True,
        #                 enable_mask = True,
        # )
        # loss = loss_dict_mask.mean()
        # loss.backward()

        xt, t, model_output, loss_dict = denoiser.training_losses(
                    transformer,
                    latents,
                    arg_c,
                    n_tokens=None,
                    i2v_mode=None,
                    cond_latents=None,
                    args=args,
                    training_cache=True,
                    enable_mask = False,
        )
        x0 = None  # not returned by training_losses
        loss = loss_dict['loss'].mean()

        loss.backward()


        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()
        wandb_log({"train/t2i_flow_loss": avg_loss.item()}, step=step)

    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if step % validation_steps == 0:
        sampling_sigmas = get_sampling_sigmas(25, 7.0)
        latent = [torch.randn_like(noise)]
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(25):
                    latent_model_input = latent

                    timestep = [sampling_sigmas[i]*1000]
                    timestep = torch.tensor(timestep).to(device)

                    noise_pred_cond = transformer(\
                    latent_model_input, t=timestep, **arg_c, flag=False)[0]
                    
                    noise_pred_uncond = transformer(\
                            latent_model_input, t=timestep, **arg_null, flag=False)[0]

                    noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)

                    if i+1 == 25:
                        latent[0] = latent[0] + (0-sampling_sigmas[i])*noise_pred_cond
                    else:
                        latent[0] = latent[0] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond
        latent = latent[0]                       
        global_step = 1
        latent = latent[:,:,:,:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = scale(vae, latent)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video_ori = scale(vae, model_input)

        t2v_dir = os.path.join(sample_output_dir, "t2v")
        os.makedirs(t2v_dir, exist_ok=True)
        gen_filename = os.path.join(
                t2v_dir,
                video_id+str(step)+"_"+"_imgt2i_"+str(device)+".mp4",
            )
        export_to_video(video[0] , gen_filename, fps=fps)
        
        ori_filename = os.path.join(
                t2v_dir,
                video_id+str(step)+"_"+"_imgorit2i_"+str(device)+".mp4",
            )
        export_to_video(video_ori[0] , ori_filename, fps=fps)

        
        txt_filename = os.path.join(
                t2v_dir,
                video_id+str(step)+"_"+"_imgorit2i_"+str(device)+".txt",
            )
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(caption)  
        if dist.get_rank() == 0 and wandb is not None and wandb.run is not None:
            try:
                wandb.log({
                    "samples_t2i/generated_video": wandb.Video(gen_filename, fps=fps, format="mp4"),
                    "samples_t2i/original_video": wandb.Video(ori_filename, fps=fps, format="mp4"),
                    "samples_t2i/caption": wandb.Html(f"<pre>{caption}</pre>"),
                }, step=step)
            except Exception:
                pass
                
    # update ema                              
    if ema_transformer is not None:
        reshard_fsdp(ema_transformer)
        for p_averaged, p_model in zip(ema_transformer.parameters(),
                                       transformer.parameters()):
            with torch.no_grad():
                p_averaged.copy_(
                    torch.lerp(p_averaged.detach(), p_model.detach(),
                               1 - ema_decay))

    return total_loss, grad_norm.item(), model_pred_norm, step1, step2

def distill_one_step(
    transformer,
    result_list,
    model_type,
    teacher_transformer,
    ema_transformer,
    optimizer,
    discriminator,
    discriminator_optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    solver,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    num_euler_timesteps,
    multiphase,
    not_apply_cfg_solver,
    distill_cfg,
    ema_decay,
    pred_decay_weight,
    pred_decay_type,
    hunyuan_teacher_disable_cfg,
    device,
    vae=None,
    text_encoder=None,
    clip = None,
    source_idx_double=None,
    source_idx_single=None,
    step=None,
    step2=None,
    wan_i2v=None,
    denoiser=None,
    camption_model = None,
    tokenizer = None,
    rank = None,
    world_size = None,
    caption_img_dir = None,
    sample_output_dir = None,
    fps = 16,
    validation_steps = 16,
):
    total_loss = 0.0
    optimizer.zero_grad()
    model_pred_norm = {
        "fro": 0.0,  # codespell:ignore
        "largest singular value": 0.0,
        "absolute mean": 0.0,
        "absolute max": 0.0,
    }
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
    negative_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
  
    for _ in range(gradient_accumulation_steps):
       
        #rand_num = random.random()  # i2v or v2v

        rand_num = random.random()  # i2v or v2v
        rank = dist.get_rank()
        rand_num = torch.ones(1)
        if rank == 0:
            rand_num = random.random()  # i2v or v2v

            s1 = torch.tensor([rand_num], dtype=torch.float32, device=device)
        else:
            s1 = torch.ones(1, dtype=torch.float32)
        s1 = s1.to(device)
        dist.broadcast(s1, src=0)
        rand_num = float(s1)
        
        (
            pixel_values_vid,
            pixel_values_ref_img,
            caption,
            K_ctrl,
            c2w_ctrl,
            videoid,
        ) = next(loader)
        latent_frame_zero = 8


        rand_num = 0.3

        frame_pixel = pixel_values_vid.shape[1]
        frame_pixel = ( frame_pixel // 4 ) * 4 + 1
        if frame_pixel > pixel_values_vid.shape[1]:
            frame_pixel = frame_pixel - 4

        pixel_values_vid = pixel_values_vid[:,:frame_pixel]



        rand_num_img = random.random()  # i2v or v2v
        if pixel_values_vid.shape[1] <= 33:
            rand_num_img = 0.3



        # Cap frame count before GPU transfer. After squeeze shape is [T, C, H, W].
        # 700 frames × 704×1280 float32 ≈ 7.2 GB, safe margin for VAE encode.
        MAX_PIXEL_FRAMES = 1000
        with torch.no_grad():
            pixel_values_vid = pixel_values_vid.squeeze()
            if pixel_values_vid.shape[0] > MAX_PIXEL_FRAMES:
                pixel_values_vid = pixel_values_vid[-MAX_PIXEL_FRAMES:].contiguous()
            pixel_values_vid = pixel_values_vid.permute(1,0,2,3).contiguous().to(device)
            pixel_values_ref_img = pixel_values_ref_img.squeeze().to(device)
            latents = pixel_values_vid 

        max_area=704 * 1280
        iw, ih  = latents.shape[2:] 
        dh, dw = wan_i2v.patch_size[1] * wan_i2v.vae_stride[1], wan_i2v.patch_size[
                2] * wan_i2v.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        scale1 = max(ow / iw, oh / ih)
        latents = F.interpolate(latents, size=(round(iw * scale1), round(ih * scale1)), mode='bilinear', align_corners=False)
        pixel_values_vid = None  # free original GPU tensor; latents holds the only needed ref

        model_input = latents
        h1,w1 = latents.shape[2:]

        rand_num_img1 = rand_num_img
        if rand_num_img < 0.4:
            model_input = model_input[:,-33:]
            model_input = torch.cat([model_input[:,0].unsqueeze(1).repeat(1,16,1,1), model_input[:,:33]],dim=1)
            rand_num_img = 0.6
            rand_num_img1 = 0.3

        # H1 fix: release large latents reference so original full tensor can be freed
        # before VAE encode. When rand_num_img < 0.4 created a new model_input via
        # torch.cat (no shared storage), this frees the interpolated full-frame tensor.
        # When model_input already equals latents (no slicing), this is a no-op.
        latents = model_input

        model_input_caption = model_input[:,-32:]

        frame = model_input.shape[1]
  
        rand_num_caption = random.random()  # i2v or v2v
        rank = dist.get_rank()
        rand_num_caption = torch.ones(1, dtype=torch.float32, device=device)
        if rank == 0:
            rand_num_caption = random.random()  # i2v or v2v
            s1 = torch.tensor([rand_num_caption], dtype=torch.float32, device=device)
        else:
            s1 = torch.ones(1, dtype=torch.float32)
        s1 = s1.to(device)
        dist.broadcast(s1, src=0)
        rand_num_caption = float(s1)
            
        if rand_num_caption > 0.4:
            img1,img2,img3,img4 = extract_first_frame_from_latents(model_input_caption[:,0:1]),extract_first_frame_from_latents(model_input_caption[:,7:8]),\
            extract_first_frame_from_latents(model_input_caption[:,15:16]),extract_first_frame_from_latents(model_input_caption[:,31:32])

            os.makedirs(caption_img_dir, exist_ok=True)
            path1 = os.path.join(caption_img_dir, f"_{step}_{rank}_1.jpg")
            img1.save(path1)
            path2 = os.path.join(caption_img_dir, f"_{step}_{rank}_2.jpg")
            img2.save(path2)
            path3 = os.path.join(caption_img_dir, f"_{step}_{rank}_3.jpg")
            img3.save(path3)
            path4 = os.path.join(caption_img_dir, f"_{step}_{rank}_4.jpg")
            img4.save(path4)

            generation_config = dict(max_new_tokens=1024, do_sample=True)

            pixel_values1 = load_image(path1, max_num=12).to(torch.bfloat16).to(device)
            pixel_values2 = load_image(path2, max_num=12).to(torch.bfloat16).to(device)
            pixel_values3 = load_image(path3, max_num=12).to(torch.bfloat16).to(device)
            pixel_values4 = load_image(path4, max_num=12).to(torch.bfloat16).to(device)
            pixel_values = torch.cat((pixel_values1, pixel_values2, pixel_values3, pixel_values4), dim=0)


            # question = '<image>\nWatch the given egocentric (first-person) image and write a detailed, content-rich caption of around 70 words for video generation, focusing only on visible people, objects, scenery, weather, lighting, atmosphere, and activities, and avoiding any mention of camera movement, lens changes, or filming techniques.'
            question = '<image>\nWatch the given egocentric (first-person) hand-object interaction image and write a detailed, content-rich caption of around 70 words for video generation, focusing on the hand actions, objects being manipulated, surrounding environment, lighting, and atmosphere, and avoiding any mention of camera movement, lens changes, or filming techniques.'

            response, history = camption_model.chat(tokenizer, pixel_values, question, generation_config,
                                           history=None, return_history=True)
            caption_ori = caption[0]
            caption = list(caption)
            caption[0] = caption[0] + " " + response

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model_input = torch.cat([wan_i2v.vae.encode([model_input.to(device)[:,:-32].to(device)])[0], \
                                             wan_i2v.vae.encode([model_input.to(device)[:,-32:].to(device)])[0]],dim=1) 
                    model_input_sample = model_input

            latents = model_input
            img = model_input[:,:-latent_frame_zero]

        else:

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model_input = torch.cat([wan_i2v.vae.encode([model_input.to(device)[:,:-32].to(device)])[0], \
                                             wan_i2v.vae.encode([model_input.to(device)[:,-32:].to(device)])[0]],dim=1) 
                    model_input_sample = model_input


            latents = model_input
            img =  model_input[:,:-latent_frame_zero]

            input_encoder = True
            if rand_num_img < 0.4:
                input_encoder = False

      
            
        with torch.no_grad():
            arg_c, arg_null, noise, mask2, img = wan_i2v.generate(
                        caption[0],
                        frame_num=frame,
                        max_area=max_area,
                        latent_frame_zero=latent_frame_zero,
                        img=img)
        mask2[0] = mask2[0].to(latents.dtype)


        
        if rand_num_caption > 0.4:  
            context1 = wan_i2v.text_encoder([caption_ori], device)  
            context2 = wan_i2v.text_encoder([response], device)  
            context = torch.cat([context1[0], context2[0]], dim=0)
            # Truncate to model's text_len to avoid negative padding dimension
            text_len = wan_i2v.model.text_len
            context = context[:text_len]
            arg_c['context'] = [context]
        
        #MVDT
        # if False:
        #     xt, t, model_output, loss_dict_mask, x0, t  = denoiser.training_losses_i2v_pack(
        #                 transformer,
        #                 latents,
        #                 arg_c,
        #                 n_tokens=None,
        #                 i2v_mode=None,
        #                 cond_latents=None,
        #                 args=args,
        #                 latent_frame_zero=latent_frame_zero,
        #                 training_cache=True,
        #                 enable_mask = True,
        #                 mask2 =  mask2,
        #                 img = img[0],
        #                 step = step,
        #     )
        #     loss = loss_dict_mask.mean()
        #     loss.backward()

        xt, t, model_output, loss_dict, x0, t  = denoiser.training_losses_i2v_pack(
                    transformer,
                    latents,
                    arg_c,
                    n_tokens=None,
                    i2v_mode=None,
                    cond_latents=None,
                    args=args,
                    latent_frame_zero=latent_frame_zero,
                    training_cache=True,
                    enable_mask = False,
                    mask2 =  mask2,
                    img = img[0],
                    step = step,
        )
        loss = loss_dict.mean()


        loss.backward()
        latent_frame_zero = 8

        grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % validation_steps == 0:

            latent = noise.detach().squeeze()
            sample_step = 25
            sampling_sigmas = get_sampling_sigmas(sample_step, 7.0)
            latent = (1. - mask2[0]) * img[0]  + mask2[0] * latent


            latent_frame_zero = 8

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(sample_step):
                        latent_model_input = [latent]

                        timestep = [sampling_sigmas[i]*1000]
                        timestep = torch.tensor(timestep).to(device)
                        temp_ts = (mask2[0][0][:-latent_frame_zero, ::2, ::2] ).flatten()
                        temp_ts = torch.cat([
                            temp_ts,
                            temp_ts.new_ones(arg_c['seq_len'] - temp_ts.size(0)) * timestep
                        ])
                        timestep = temp_ts.unsqueeze(0)

                        noise_pred_cond = transformer(\
                                latent_model_input, t=timestep, latent_frame_zero=latent_frame_zero, **arg_c)[0]
                        noise_pred_uncond = transformer(\
                                latent_model_input, t=timestep, latent_frame_zero=latent_frame_zero, **arg_null)[0]

                        noise_pred_cond = noise_pred_uncond + 5.0*(noise_pred_cond - noise_pred_uncond)
                        if i+1 == sample_step:
                            temp_x0 = latent[:,-latent_frame_zero:,:,:] + (0-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]
                        else:
                            temp_x0 = latent[:,-latent_frame_zero:,:,:] + (sampling_sigmas[i+1]-sampling_sigmas[i])*noise_pred_cond[:,-latent_frame_zero:,:,:]

                        latent = torch.cat([model_input_sample[:,:-latent_frame_zero,:,:], temp_x0], dim=1)
                        # print(latent.shape, img[0].shape,temp_x0.shape,"dxh98d1")
                        latent = (1. - mask2[0]) * img[0]  + mask2[0] * latent


            global_step = 1
            latent = latent[:,-latent_frame_zero:,:,:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = scale(vae, latent)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video_ori = scale(vae, model_input[:,-latent_frame_zero:,:,:]) #[:,-latent_frame_zero:,:,:])
                
            # Ensure videoid is a string
            if isinstance(videoid, list):
                videoid_str = "_".join(map(str, videoid))
            else:
                videoid_str = str(videoid)

            i2v_dir = os.path.join(sample_output_dir, "i2v")
            os.makedirs(i2v_dir, exist_ok=True)
            if rand_num_img1 < 0.4:
                gen_filename = os.path.join(
                                    i2v_dir,
                                    videoid_str+"_"+"_i2vnormnew_img_2_"+str(device)+".mp4",
                                )
                export_to_video(video[0] , gen_filename, fps=fps)
                ori_filename = os.path.join(
                                    i2v_dir,
                                    videoid_str+"_"+"_i2vnormnewori_img_2_"+str(device)+".mp4",
                                )
                export_to_video(video_ori[0] , ori_filename, fps=fps)
            else:
                gen_filename = os.path.join(
                                    i2v_dir,
                                    videoid_str+"_"+"_normnew_2_"+str(device)+".mp4",
                                )
                export_to_video(video[0] , gen_filename, fps=fps)
                ori_filename = os.path.join(
                                    i2v_dir,
                                    videoid_str+"_"+"_normnew_ori_2_"+str(device)+".mp4",
                                )
                export_to_video(video_ori[0] , ori_filename, fps=fps)
            txt_filename = os.path.join(
                                    i2v_dir,
                                    videoid_str+"_"+"_i2vnormnew_img_2_"+str(device)+".txt",
                                )
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(caption[0])
            if dist.get_rank() == 0 and wandb is not None and wandb.run is not None:
                try:
                    wandb.log({
                        "samples_i2v/generated_video": wandb.Video(gen_filename, fps=fps, format="mp4"),
                        "samples_i2v/original_video": wandb.Video(ori_filename, fps=fps, format="mp4"),
                        "samples_i2v/caption": wandb.Html(f"<pre>{caption[0]}</pre>"),
                    }, step=step)
                except Exception:
                    pass
            del video, video_ori, latent
            torch.cuda.empty_cache()
            gc.collect()
        import torchvision.io

        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        print(dist.get_rank(),dist.get_world_size(),avg_loss.item(),"avg_loss")
        total_loss += avg_loss.item()
        wandb_log({"train/i2v_flow_loss": avg_loss.item()}, step=step)

    # update ema
    if ema_transformer is not None:
        reshard_fsdp(ema_transformer)
        for p_averaged, p_model in zip(ema_transformer.parameters(),
                                       transformer.parameters()):
            with torch.no_grad():
                p_averaged.copy_(
                    torch.lerp(p_averaged.detach(), p_model.detach(),
                               1 - ema_decay))


    #wan_i2v.text_encoder.model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss, grad_norm.item(), model_pred_norm, step2


import wan23
from wan23.configs import WAN_CONFIGS

import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample_conv3d_weights(conv_small,size):
    old_weight = conv_small.weight.data 
    new_weight = F.interpolate(
        old_weight,                      # 输入张量
        size=size,                  # 目标尺寸（时间维度不变）
        mode='trilinear',                # 3D插值
        align_corners=False              # 避免边缘对齐伪影
    )
    conv_large = nn.Conv3d(
        in_channels=16,
        out_channels=5120,
        kernel_size=size,
        stride=size,
        padding=0
    )
    conv_large.weight.data = new_weight
    # 如果有偏置项，直接复制（无需修改）
    if conv_small.bias is not None:
        conv_large.bias.data = conv_small.bias.data.clone()
    return conv_large

import datetime

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=36000))
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # Append a timestamp subdirectory so every run gets its own folder
    if args.output_dir is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Broadcast the timestamp string from rank-0 so all ranks use the same path
        ts_tensor = torch.zeros(15, dtype=torch.uint8, device=device)
        if rank == 0:
            for i, c in enumerate(timestamp):
                ts_tensor[i] = ord(c)
        dist.broadcast(ts_tensor, src=0)
        timestamp = "".join(chr(int(v)) for v in ts_tensor.tolist())
        args.output_dir = os.path.join(args.output_dir, timestamp)

    # Derive default subdirectory paths from output_dir if not explicitly set
    if args.output_dir is not None:
        if args.caption_img_dir is None:
            args.caption_img_dir = os.path.join(args.output_dir, "caption_imgs")
        if args.sample_output_dir is None:
            args.sample_output_dir = os.path.join(args.output_dir, "samples")

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0 and args.use_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Create model:
    cfg = WAN_CONFIGS["ti2v-5B"]
    ckpt_dir = "./Yume-5B-720P"

    # Referenced from https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py
    wan_i2v = wan23.Yume(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
    )  
    transformer = wan_i2v.model   
    transformer = transformer.train().requires_grad_(True)
    
    
    if args.resume_from_checkpoint:
        (
            transformer,
            init_steps,
        ) = resume_checkpoint(
            transformer,
            args.resume_from_checkpoint,
        )

    from ADD.models.discriminator import ProjectedDiscriminator
    discriminator = ProjectedDiscriminator(c_dim=384).train()

    if args.use_ema:
        ema_transformer = deepcopy(transformer)
    else:
        ema_transformer = None

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
        rank=device,
    )
    discriminator_fsdp_kwargs = get_discriminator_fsdp_kwargs(
         args.master_weight_type)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        # WAN model attention projections are named q/k/v/o (not to_k/to_q etc.)
        lora_target_modules = ["q", "k", "v", "o"]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()

        # Store LoRA config on the underlying model config for checkpoint saving
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = lora_target_modules

        # peft's fsdp_auto_wrap_policy needs _no_split_modules as class-name strings
        transformer._no_split_modules = [cls.__name__ for cls in no_split_modules]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](
            transformer)
    else:
        fsdp_kwargs["use_orig_params"] = True

    transformer = FSDP(
        transformer,
        **fsdp_kwargs,
    )


    discriminator = FSDP(
         discriminator,
         **discriminator_fsdp_kwargs,
         use_orig_params=True,
    )

    main_print("--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules,
                                 args.selective_checkpointing)
        if args.use_ema:
            apply_fsdp_checkpointing(ema_transformer, no_split_modules,
                                     args.selective_checkpointing)
    # Set model as trainable.
    transformer.train()
    if args.use_ema:
        ema_transformer.requires_grad_(False)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)
    if args.scheduler_type == "pcm_linear_quadratic":
        linear_steps = int(noise_scheduler.config.num_train_timesteps *
                           args.linear_range)
        sigmas = linear_quadratic_schedule(
            noise_scheduler.config.num_train_timesteps,
            args.linear_quadratic_threshold,
            linear_steps,
        )
        sigmas = torch.tensor(sigmas).to(dtype=torch.float32)
    else:
        sigmas = noise_scheduler.sigmas
    solver = EulerSolver(
        sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
    )
    solver.to(device)
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = bnb.optim.Adam8bit(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
     )
    # optimizer = torch.optim.AdamW(
    #     params_to_optimize,
    #     lr=args.learning_rate,
    #     betas=(0.9, 0.999),
    #     weight_decay=args.weight_decay,
    #     eps=1e-8,
    #  )

    params_to_optimize_dis = discriminator.parameters()
    params_to_optimize_dis = list(
        filter(lambda p: p.requires_grad, params_to_optimize_dis))

    discriminator_optimizer = bnb.optim.Adam8bit(
        params_to_optimize_dis,
        lr=args.discriminator_learning_rate,
        betas=(0, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
     )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")
    init_steps_opt = 0
    # todo add lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps_opt - 1,
    )
    train_dataset = StableVideoAnimationDataset(
        height=704, 
        width=1280, 
        n_sample_frames=33, 
        sample_rate=1,
        root_dir=args.root_dir,
        full_mp4=args.full_mp4,
    )

    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=video_collate_function,
        pin_memory=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    val_dataloader = train_dataloader

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps *
        args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)



    # Train!
    total_batch_size = (world_size * args.gradient_accumulation_steps /
                        args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(
        f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError(
            "resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )
    loader_val = sp_parallel_dataloader_wrapper(
        val_dataloader,
        device,
        world_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    init_steps = 0
    if init_steps > 0:
        train_dataloader.dataset.skip = True
        # todo future
        for i in range(init_steps):
            _ = next(loader)
        train_dataloader.dataset.skip = False

    vae = wan_i2v.vae

    dist.barrier()
    transformer.guidance_embed = False
    transformer_tea = None
    
    wan_i2v.device = device
    denoiser = load_denoiser()

    wan_i2v.text_encoder.model.to(torch.bfloat16)
    fsdp_kwargs = get_DINO_fsdp_kwargs()
    wan_i2v.text_encoder.model = FSDP(
        wan_i2v.text_encoder.model,
        **fsdp_kwargs,
        use_orig_params=True,
    )

    path = './InternVL3-2B-Instruct' # Or 78B
    camption_model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    

    prompt_all = []
    result_list = []  # Initialize result_list
    text_encoder = wan_i2v.text_encoder  # Use text encoder from wan_i2v model
    fps = args.fps

    step1 = 0
    step2 = 0
    step3 = 0

    for step in range(init_steps + 1, args.max_train_steps + 1):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        if step%2 == 0:
            loss, grad_norm, pred_norm, step2 = distill_one_step(
                transformer,
                result_list,
                args.model_type,
                transformer_tea,
                ema_transformer,
                optimizer,
                discriminator,
                discriminator_optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                solver,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.max_grad_norm,
                args.num_euler_timesteps,
                1,
                args.not_apply_cfg_solver,
                args.distill_cfg,
                args.ema_decay,
                args.pred_decay_weight,
                args.pred_decay_type,
                args.hunyuan_teacher_disable_cfg,
                device,
                vae = vae,
                text_encoder = None,
                clip = None,
                source_idx_double = args.source_idx_double,
                source_idx_single = args.source_idx_single,
                step = step,
                step2 = step2,
                wan_i2v = wan_i2v,
                denoiser = denoiser,
                camption_model = camption_model,
                tokenizer = tokenizer,
                rank = rank,
                world_size = world_size,
                caption_img_dir = args.caption_img_dir,
                sample_output_dir = args.sample_output_dir,
                fps = args.fps,
                validation_steps = args.validation_steps,
            )
        else:
            loss, grad_norm, pred_norm, step1, step2 = distill_one_step_t2i(
                transformer,
                result_list,
                prompt_all,
                args.model_type,
                transformer_tea,
                ema_transformer,
                optimizer,
                discriminator,
                discriminator_optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                solver,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.max_grad_norm,
                args.num_euler_timesteps,
                1,
                args.not_apply_cfg_solver,
                args.distill_cfg,
                args.ema_decay,
                args.pred_decay_weight,
                args.pred_decay_type,
                args.hunyuan_teacher_disable_cfg,
                device,
                vae = vae,
                text_encoder = None,
                clip = None,
                source_idx_double = args.source_idx_double,
                source_idx_single = args.source_idx_single,
                step = step,
                step1 = step1,
                step2=step2,
                wan_i2v = wan_i2v,
                denoiser = denoiser,
                camption_model = camption_model,
                tokenizer = tokenizer,
                rank = rank,
                world_size=world_size,
                caption_img_dir = args.caption_img_dir,
                sample_output_dir = args.sample_output_dir,
                fps = args.fps,
                validation_steps = args.validation_steps,
            )
        

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": grad_norm,
        })
        progress_bar.update(1)
        wandb_log({
            "train/total_loss": loss,
            "train/grad_norm": grad_norm,
            "train/lr": lr_scheduler.get_last_lr()[0],
            "train/step_time": step_time,
            "train/is_t2i_step": 1 if step % 2 != 0 else 0,
        }, step=step)

        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank,
                                     args.output_dir, step)
            else:
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                # Your existing checkpoint saving code
                if args.use_ema:
                    save_checkpoint(ema_transformer, rank, args.output_dir,
                                    step)
                else:
                    save_checkpoint(transformer, rank, args.output_dir, step)
            dist.barrier()
        if args.log_validation and step % args.validation_steps == 0:
            optimizer.zero_grad()
            log_validation(
                args,
                transformer,
                device,
                torch.bfloat16,
                step,
                scheduler_type=args.scheduler_type,
                shift=args.shift,
                num_euler_timesteps=args.num_euler_timesteps,
                linear_quadratic_threshold=args.linear_quadratic_threshold,
                linear_range=args.linear_range,
                ema=False,
                loader_val=loader_val,
                vae = vae,
                text_encoder = text_encoder,
                fps=fps,
            )
            if args.use_ema:
                log_validation(
                    args,
                    ema_transformer,
                    device,
                    torch.bfloat16,
                    step,
                    scheduler_type=args.scheduler_type,
                    shift=args.shift,
                    num_euler_timesteps=args.num_euler_timesteps,
                    linear_quadratic_threshold=args.linear_quadratic_threshold,
                    linear_range=args.linear_range,
                    ema=True,
                    loader_val=loader_val,
                    vae = vae,
                    text_encoder = text_encoder,
                    fps=fps,
                )
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

    if args.use_lora:
        save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                             args.max_train_steps)
    else:
        save_checkpoint(transformer, rank, args.output_dir,
                        args.max_train_steps)

    if rank == 0 and wandb is not None and wandb.run is not None:
        wandb.finish()

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type",
                        type=str,
                        default="mochi",
                        help="The type of model to train.")
    # dataset & dataloader
    # parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    # parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str)
    parser.add_argument("--model_vae_path", type=str)
    parser.add_argument("--model_text_emb", type=str)
    # parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--validation_sampling_steps", type=str, default="64")
    parser.add_argument("--validation_guidance_scale", type=str, default="4.5")

    parser.add_argument("--validation_steps", type=float, default=64)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing per-video frame folders (mp4_frame).",
    )
    parser.add_argument(
        "--full_mp4",
        type=str,
        required=True,
        help="Directory containing the full MP4 video files.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "Root directory for all outputs. Each run creates a timestamped subdirectory here.",
    )
    parser.add_argument(
        "--caption_img_dir",
        type=str,
        default=None,
        help="Directory for temporary captioning images saved during training. "
             "Defaults to <output_dir>/caption_imgs.",
    )
    parser.add_argument(
        "--sample_output_dir",
        type=str,
        default=None,
        help="Directory for training-step preview videos. "
             "Defaults to <output_dir>/samples.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for exported preview videos.",
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
         " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--discriminator_learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial discriminator learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help=
        "Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size",
                        type=int,
                        default=1,
                        help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=256,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank",
                        type=int,
                        default=128,
                        help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--not_apply_cfg_solver",
        action="store_true",
        help="Whether to apply the cfg_solver.",
    )
    parser.add_argument("--distill_cfg",
                        type=float,
                        default=3.0,
                        help="Distillation coefficient.")
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument("--scheduler_type",
                        type=str,
                        default="pcm",
                        help="The scheduler type to use.")
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="Threshold for linear quadratic scheduler.",
    )
    parser.add_argument(
        "--linear_range",
        type=float,
        default=0.5,
        help="Range for linear quadratic scheduler.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.001,
                        help="Weight decay to apply.")
    parser.add_argument("--use_ema",
                        action="store_true",
                        help="Whether to use EMA.")
    parser.add_argument("--t5_cpu", action="store_true",
                        help="Whether to place T5 model on CPU.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="yume-stage1", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--multi_phased_distill_schedule",
                        type=str,
                        default=None)
    parser.add_argument("--pred_decay_weight", type=float, default=0.0)
    parser.add_argument("--pred_decay_type", default="l1")
    parser.add_argument("--source_idx_double", nargs='+', type=int)
    parser.add_argument("--source_idx_single", nargs='+', type=int)
    parser.add_argument("--hunyuan_teacher_disable_cfg", action="store_true")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)
