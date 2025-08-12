# -*- coding: utf-8 -*-
# ==============================================================================
# arxive: https://arxiv.org/abs/2507.03905
# GitHUb: https://github.com/antgroup/echomimic_v3
# Project Page: https://antgroup.github.io/ai/echomimic_v3/
# ==============================================================================

import os
import math
import datetime
import sys
import subprocess
from functools import partial

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from moviepy import VideoFileClip, AudioFileClip
import librosa

# Custom modules
from diffusers import FlowMatchEulerDiscreteScheduler

from src.dist import set_multi_gpus_devices
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import  CLIPModel
from src.wan_text_encoder import  WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline

from src.utils import (
    filter_kwargs,
    get_image_to_video_latent3,
    save_videos_grid,
)
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients

from src.face_detect import get_mask_coord

import argparse
import gradio as gr
import random
import gc

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7860, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()


# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        # General settings
        self.ulysses_degree = 1
        self.ring_degree = 1
        self.fsdp_dit = False

        # Pipeline parameters
        self.num_skip_start_steps = 5
        self.teacache_offload = True
        self.cfg_skip_ratio = 0
        self.enable_riflex = False
        self.riflex_k = 6

        # Paths
        self.config_path = "config/config.yaml"
        self.model_name = "models/Wan2.1-Fun-V1.1-1.3B-InP"
        self.transformer_path = "models/transformer/diffusion_pytorch_model.safetensors"
        # Performance
        self.cpu_offload = True  # Offload models between CPU/GPU to reduce VRAM
        self.offload_mode = "model"  # "sequential" (lowest VRAM), "model" (balanced), "none" (max VRAM)

        # Sampler and audio settings
        self.sampler_name = "Flow_DPM++"
        self.audio_scale = 1.0
        self.enable_teacache = True
        self.teacache_threshold = 0.1
        self.num_skip_start_steps = 5
        self.shift = 5.0
        self.use_un_ip_mask = False

        self.partial_video_length = 65
        self.overlap_video_length = 4
        self.neg_scale = 1.5
        self.neg_steps = 2
        self.guidance_scale = 4.5 #3.0 ~ 6.0
        self.audio_guidance_scale = 2.5 #2.0 ~ 3.0
        self.use_dynamic_cfg = True
        self.use_dynamic_acfg = True
        self.seed = 43
        self.num_inference_steps = 12
        self.lora_weight = 1.0

        # Model settings
        self.weight_dtype = torch.bfloat16
        self.sample_size = [448, 448]
        self.fps = 25

        # Test data paths
        self.wav2vec_model_dir = "models/wav2vec2-base-960h"


# --------------------- Helper Functions ---------------------
def ensure_package(pkg: str, spec: str | None = None):
    try:
        __import__(pkg)
    except ImportError:
        to_install = spec if spec is not None else pkg
        print(f"Installing dependency: {to_install} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", to_install])


def snapshot(repo_id: str, local_dir: str, allow_patterns=None):
    """Download a repo snapshot into local_dir (no symlinks)."""
    ensure_package("huggingface_hub", "huggingface_hub>=0.24.0")
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading '{repo_id}' into '{local_dir}' ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        repo_type="model",
    )


def ensure_models_available(config, cfg):
    """Ensure all required model assets exist locally; download if missing."""
    transformer_subpath = os.path.normpath(
        cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')
    )
    base_dir = config.model_name
    transformer_dir = base_dir if transformer_subpath in (".", "") else os.path.join(base_dir, transformer_subpath)
    config_json_path = os.path.join(transformer_dir, "config.json")
    if not os.path.isfile(config_json_path):
        snapshot("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", base_dir)

    transformer_file = config.transformer_path
    if transformer_file and not os.path.isfile(transformer_file):
        snapshot(
            "BadToBest/EchoMimicV3",
            os.path.join("models"),
            allow_patterns=["transformer/*"],
        )

    wav2vec_dir = config.wav2vec_model_dir
    # We check for either tokenizer.json or config.json presence to consider it downloaded
    if not (os.path.isdir(wav2vec_dir) and (
        os.path.isfile(os.path.join(wav2vec_dir, "config.json")) or
        os.path.isfile(os.path.join(wav2vec_dir, "tokenizer.json"))
    )):
        snapshot("facebook/wav2vec2-base-960h", wav2vec_dir)

    return transformer_dir
def load_wav2vec_models(wav2vec_model_dir):
    """Load Wav2Vec models for audio feature extraction."""
    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_dir)
    # Keep Wav2Vec2 on CPU to save VRAM; audio features are small and moved later
    model = Wav2Vec2Model.from_pretrained(wav2vec_model_dir).eval()
    model.requires_grad_(False)
    return processor, model


def extract_audio_features(audio_path, processor, model):
    """Extract audio features using Wav2Vec."""
    sr = 16000
    audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
    input_values = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
    # Run on CPU to save VRAM
    with torch.no_grad():
        features = model(input_values).last_hidden_state
    return features.squeeze(0)


def get_sample_size(image, default_size):
    """Calculate the sample size based on the input image dimensions."""
    width, height = image.size
    original_area = width * height
    default_area = default_size[0] * default_size[1]

    if default_area < original_area:
        ratio = math.sqrt(original_area / default_area)
        width = width / ratio // 16 * 16
        height = height / ratio // 16 * 16
    else:
        width = width // 16 * 16
        height = height // 16 * 16

    return int(height), int(width)


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()


# Initialize configuration
config = Config()

# Set up multi-GPU devices
device = set_multi_gpus_devices(config.ulysses_degree, config.ring_degree)

# Load configuration file
cfg = OmegaConf.load(config.config_path)

# Load models
# Ensure required models are present (download if missing)
transformer_root = ensure_models_available(config, cfg)

transformer = WanTransformerAudioMask3DModel.from_pretrained(
    transformer_root,
    transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
    torch_dtype=config.weight_dtype,
)
if config.transformer_path is not None:
    if config.transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(config.transformer_path)
    else:
        state_dict = torch.load(config.transformer_path)
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(config.model_name, cfg['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
).to(config.weight_dtype)

tokenizer = AutoTokenizer.from_pretrained(
    (
        (lambda: (
            os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
            if os.path.isdir(os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
            else cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')
        ))()
    ),
)

text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(config.model_name, cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
    torch_dtype=config.weight_dtype,
).eval()

clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(config.model_name, cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(config.weight_dtype).eval()

# Load scheduler
scheduler_cls = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[config.sampler_name]
scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(cfg['scheduler_kwargs'])))

# Create pipeline
pipeline = WanFunInpaintAudioPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
)

def set_offload_mode(pipeline, mode: str, device):
    try:
        getattr(pipeline, "maybe_free_model_hooks", lambda *a, **k: None)()
    except Exception:
        pass
    try:
        if mode == "none":
            # Fully disable offload and keep pipeline on GPU
            pipeline.to(device=device)
        elif mode == "sequential":
            # Move to CPU first, then enable sequential offload
            pipeline.to("cpu")
            getattr(pipeline, "enable_sequential_cpu_offload", lambda *a, **k: None)()
        elif mode == "model":
            # Move to CPU first, then enable model offload
            pipeline.to("cpu")
            getattr(pipeline, "enable_model_cpu_offload", lambda *a, **k: None)()
        elif mode == "hybrid24":
            # Hybrid for ~24GB GPUs: keep heavy modules on GPU, offload the rest
            # Start clean
            getattr(pipeline, "maybe_free_model_hooks", lambda *a, **k: None)()
            # Put everything on CPU first
            pipeline.to("cpu")
            # Move the heaviest modules to GPU
            try:
                pipeline.transformer.to(device=device)
            except Exception:
                pass
            # Keep VAE on CPU to avoid bf16 3D conv slow path issues on some CUDA builds
            # Keep text and clip encoders on CPU to save VRAM; they run once per run
            # Enable slicing to reduce peak inside GPU modules
            with torch.no_grad():
                getattr(pipeline, "enable_attention_slicing", lambda *a, **k: None)("max")
                getattr(pipeline, "enable_vae_slicing", lambda *a, **k: None)()
                getattr(pipeline, "enable_xformers_memory_efficient_attention", lambda *a, **k: None)()
        with torch.no_grad():
            getattr(pipeline, "enable_attention_slicing", lambda *a, **k: None)("max")
            getattr(pipeline, "enable_vae_slicing", lambda *a, **k: None)()
            getattr(pipeline, "enable_xformers_memory_efficient_attention", lambda *a, **k: None)()
    except Exception:
        pipeline.to(device=device)

# Do not apply offload policy until generation; avoids duplicate setup and warnings when switching modes

# Enable TeaCache if required
if config.enable_teacache:
    coefficients = get_teacache_coefficients(config.model_name)
    pipeline.transformer.enable_teacache(
        coefficients, config.num_inference_steps, config.teacache_threshold,
        num_skip_start_steps=config.num_skip_start_steps, offload=config.teacache_offload
    )

# Load Wav2Vec models
wav2vec_processor, wav2vec_model = load_wav2vec_models(config.wav2vec_model_dir)


def generate(
    image,
    audio,
    prompt,
    negative_prompt,
    seed_param,
    ui_max_side,
    ui_steps,
    ui_target_frames,
    ui_offload_mode,
):
    if seed_param<0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)

    # Process test cases
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Load reference image and prompt
    ref_img = Image.open(image).convert("RGB")

    y1, y2, x1, x2, h_, w_ = get_mask_coord(image)

    # Extract audio features
    audio_clip = AudioFileClip(audio)
    audio_features = extract_audio_features(audio, wav2vec_processor, wav2vec_model)
    # Keep audio embeds on CPU until needed; they'll be moved inside the pipeline forward
    audio_embeds = audio_features.unsqueeze(0).to(dtype=config.weight_dtype)

    # Calculate video length and latent frames
    video_length = int(audio_clip.duration * config.fps)
    if ui_target_frames is not None and ui_target_frames > 0:
        video_length = min(video_length, int(ui_target_frames))
    video_length = (
        int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if config.enable_riflex:
        pipeline.transformer.enable_riflex(k = config.riflex_k, L_test = latent_frames)

    # Allow runtime offload selection (configure once per call)
    set_offload_mode(pipeline, ui_offload_mode, device)

    # Adjust sample size by max side (keep aspect ratio, downscale only, snap to 16)
    if ui_max_side and int(ui_max_side) > 0:
        orig_w, orig_h = ref_img.size
        max_side = int(ui_max_side)
        scale = 1.0 if max_side >= max(orig_w, orig_h) else max_side / max(orig_w, orig_h)
        sample_width = max(16, int((orig_w * scale) // 16 * 16))
        sample_height = max(16, int((orig_h * scale) // 16 * 16))
    else:
        sample_height, sample_width = get_sample_size(ref_img, config.sample_size)
    downratio = math.sqrt(sample_height * sample_width / h_ / w_)
    coords = (
        y1 * downratio // 16, y2 * downratio // 16,
        x1 * downratio // 16, x2 * downratio // 16,
        sample_height // 16, sample_width // 16,
    )
    ip_mask = get_ip_mask(coords).unsqueeze(0)
    ip_mask = torch.cat([ip_mask]*3).to(device=device, dtype=config.weight_dtype)

    partial_video_length = int((config.partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (partial_video_length - 1) // vae.config.temporal_compression_ratio + 1

    # get clip image
    _, _, clip_image = get_image_to_video_latent3(ref_img, None, video_length=partial_video_length, sample_size=[sample_height, sample_width])

    # Generate video in chunks
    init_frames = 0
    last_frames = init_frames + partial_video_length
    new_sample = None

    while init_frames < video_length:
        if last_frames >= video_length:
            partial_video_length = video_length - init_frames
            partial_video_length = (
                int((partial_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
                if video_length != 1 else 1
            )
            latent_frames = (partial_video_length - 1) // vae.config.temporal_compression_ratio + 1

            if partial_video_length <= 0:
                break

        input_video, input_video_mask, _ = get_image_to_video_latent3(
            ref_img, None, video_length=partial_video_length, sample_size=[sample_height, sample_width]
        )

        partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + partial_video_length) * 2]
        # video_length = init_frames + partial_video_length

        sample = pipeline(
            prompt,
            num_frames            = partial_video_length,
            negative_prompt       = negative_prompt,
            audio_embeds          = partial_audio_embeds,
            audio_scale           = config.audio_scale,
            ip_mask               = ip_mask,
            use_un_ip_mask        = config.use_un_ip_mask,
            height                = sample_height,
            width                 = sample_width,
            generator             = generator,
            neg_scale             = config.neg_scale,
            neg_steps             = config.neg_steps,
            use_dynamic_cfg       = config.use_dynamic_cfg,
            use_dynamic_acfg      = config.use_dynamic_acfg,
            guidance_scale        = config.guidance_scale,
            audio_guidance_scale  = config.audio_guidance_scale,
            num_inference_steps   = int(ui_steps) if ui_steps and ui_steps > 0 else config.num_inference_steps,
            video                 = input_video,
            mask_video            = input_video_mask,
            clip_image            = clip_image,
            cfg_skip_ratio        = config.cfg_skip_ratio,
            shift                 = config.shift,
        ).videos
        
        if init_frames != 0:
            mix_ratio = torch.from_numpy(
                np.array([float(i) / float(config.overlap_video_length) for i in range(config.overlap_video_length)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            new_sample[:, :, -config.overlap_video_length:] = (
                new_sample[:, :, -config.overlap_video_length:] * (1 - mix_ratio) +
                sample[:, :, :config.overlap_video_length] * mix_ratio
            )
            new_sample = torch.cat([new_sample, sample[:, :, config.overlap_video_length:]], dim=2)
            sample = new_sample
        else:
            new_sample = sample

        if last_frames >= video_length:
            break

        ref_img = [
            Image.fromarray(
                (sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
            ) for i in range(-config.overlap_video_length, 0)
        ]

        init_frames += partial_video_length - config.overlap_video_length
        last_frames = init_frames + partial_video_length

    # Save generated video
    video_path = os.path.join(save_path, f"{timestamp}.mp4")
    video_audio_path = os.path.join(save_path, f"{timestamp}_audio.mp4")

    print ("final sample shape:",sample.shape)
    video_length = sample.shape[2]
    print ("final length:",video_length)

    save_videos_grid(sample[:, :, :video_length], video_path, fps=config.fps)

    video_clip = VideoFileClip(video_path)
    audio_clip = audio_clip.subclipped(0, video_length / config.fps)
    video_clip = video_clip.with_audio(audio_clip)
    video_clip.write_videofile(video_audio_path, codec="libx264", audio_codec="aac", threads=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return video_audio_path, seed


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">EchoMimicV3</h2>
            </div>
            """)
    with gr.TabItem("EchoMimicV3"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Image", type="filepath", height=400)
                audio = gr.Audio(label="Audio", type="filepath")
                prompt = gr.Textbox(label="Prompt", value="")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作.")
                with gr.Row():
                    ui_max_side = gr.Slider(256, 1024, value=max(config.sample_size), step=16, label="Max side (keeps aspect ratio)")
                with gr.Row():
                    ui_steps = gr.Slider(4, 50, value=config.num_inference_steps, step=1, label="Steps")
                    ui_target_frames = gr.Slider(1, 256, value=65, step=1, label="Target Frames (0=auto)")
                ui_offload_mode = gr.Dropdown(choices=["none", "model", "sequential", "hybrid24"], value=config.offload_mode, label="Memory Policy")
                seed_param = gr.Number(label="Seed (-1 random)", value=-1)
                generate_button = gr.Button("🎬 Generate", variant='primary')
            with gr.Column():
                video_output = gr.Video(label="Result", interactive=False)
                seed_output = gr.Textbox(label="Seed")

        gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate,
        inputs = [
            image,
            audio,
            prompt,
            negative_prompt,
            seed_param,
            ui_max_side,
            ui_steps,
            ui_target_frames,
            ui_offload_mode,
        ],
        outputs = [video_output, seed_output]
    )
        

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=False
    )
