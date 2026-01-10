import argparse
import logging
import os
import sys
import threading
import time
import tempfile
import shutil
import json
import base64
import io
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import torch.distributed as dist
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool

# Global state
model = None
prompt_expander = None
rank = 0
device = 0
args_config = None
generation_lock = threading.Lock()

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def load_model(args):
    global model, prompt_expander, rank, device, args_config
    args_config = args
    
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "FSDP not supported in non-distributed env."
        assert not (args.ulysses_size > 1), "Sequence parallel not supported in non-distributed env."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size must equal world_size."
        init_distributed_group()

    # Prompt Expander
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=True) # Assume VL capability for simplicity
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=True,
                device=rank)

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Loading model for task: {args.task}")

    # Initialize Model based on task
    if "t2v" in args.task:
        model = wan.WanT2V(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank,
            t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp, use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu, convert_model_dtype=args.convert_model_dtype,
        )
    elif "ti2v" in args.task:
        model = wan.WanTI2V(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank,
            t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp, use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu, convert_model_dtype=args.convert_model_dtype,
        )
    elif "animate" in args.task:
        model = wan.WanAnimate(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank,
            t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp, use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu, convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora
        )
    elif "s2v" in args.task:
        model = wan.WanS2V(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank,
            t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp, use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu, convert_model_dtype=args.convert_model_dtype,
        )
    else: # i2v
        model = wan.WanI2V(
            config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device, rank=rank,
            t5_fsdp=args.t5_fsdp, dit_fsdp=args.dit_fsdp, use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu, convert_model_dtype=args.convert_model_dtype,
        )
    logging.info("Model loaded successfully.")

    # If worker process, enter wait loop
    if rank != 0:
        worker_loop()

def worker_loop():
    """Loop for non-rank-0 processes to wait for generation commands."""
    while True:
        # Broadcast container
        objs = [None]
        dist.broadcast_object_list(objs, src=0)
        cmd = objs[0]
        
        if cmd is None: # Exit signal
            break
        
        kwargs = cmd
        try:
            model.generate(**kwargs)
        except Exception as e:
            logging.error(f"Worker error: {e}")
        
        # Sync after generation
        if dist.is_initialized():
            dist.barrier()

app = FastAPI()

@app.post("/generate")
async def generate_endpoint(
    prompt: str = Form(...),
    size: str = Form("1280*720"),
    frame_num: int = Form(None),
    sample_steps: int = Form(None),
    sample_shift: float = Form(None),
    sample_guide_scale: float = Form(5.0),
    seed: int = Form(-1),
    offload_model: bool = Form(True),
    # 2-pass args
    final_sample_steps: int = Form(None),
    downscale: int = Form(None),
    final_window_size: int = Form(None),
    final_threshold: float = Form(None),
    kernel: int = Form(None),
    blur: float = Form(None),
    noise_add: int = Form(None),
    # Files
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    global model, rank
    
    if rank != 0:
        return {"error": "Not master node"}

    with generation_lock:
        logging.info(f"Received generation request. Prompt: {prompt}")
        
        # Process inputs
        img_pil = None
        if image:
            content = await image.read()
            img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        
        audio_path = None
        temp_audio_file = None
        if audio:
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1])
            content = await audio.read()
            temp_audio_file.write(content)
            temp_audio_file.close()
            audio_path = temp_audio_file.name

        # Prompt extension
        if args_config.use_prompt_extend and prompt_expander:
            logging.info("Extending prompt...")
            res = prompt_expander(prompt, image=img_pil, tar_lang=args_config.prompt_extend_target_lang, seed=seed)
            if res.status:
                prompt = res.prompt
                logging.info(f"Extended prompt: {prompt}")

        # Prepare kwargs for model.generate
        cfg = WAN_CONFIGS[args_config.task]
        
        # Common args
        kwargs = {
            "seed": seed,
            "offload_model": offload_model,
            "guide_scale": sample_guide_scale,
            "n_prompt": args_config.sample_neg_prompt if hasattr(args_config, 'sample_neg_prompt') else "",
        }

        # Task specific args
        if "t2v" in args_config.task:
            kwargs.update({
                "input_prompt": prompt,
                "size": SIZE_CONFIGS[size],
                "frame_num": frame_num or cfg.frame_num,
                "shift": sample_shift or cfg.sample_shift,
                "sampling_steps": sample_steps or cfg.sample_steps,
                "sample_solver": 'unipc',
            })
        elif "i2v" in args_config.task: # Includes WanI2V and WanTI2V if using image
            kwargs.update({
                "input_prompt": prompt,
                "img": img_pil,
                "max_area": MAX_AREA_CONFIGS[size],
                "frame_num": frame_num or cfg.frame_num,
                "shift": sample_shift or cfg.sample_shift,
                "sampling_steps": sample_steps or cfg.sample_steps,
                "sample_solver": 'unipc',
                # 2-pass specific
                "final_sampling_steps": final_sample_steps,
                "downscale": downscale,
                "final_window_size": final_window_size,
                "final_threshold": final_threshold,
                "kernel": kernel,
                "blur": blur,
                "noise_add": noise_add,
            })
        elif "s2v" in args_config.task:
             kwargs.update({
                "input_prompt": prompt,
                "ref_image_path": None, # Handle PIL directly in s2v? s2v expects path usually, need check
                # WanS2V expects ref_image_path string usually, but let's see if we can hack it or if we need to save img_pil
                "audio_path": audio_path,
                # ... other s2v args ...
            })
             # For S2V, the code expects paths. We might need to save img_pil to temp file.
             if img_pil:
                 t_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                 img_pil.save(t_img.name)
                 t_img.close()
                 kwargs["ref_image_path"] = t_img.name

        # Distributed Broadcast
        if dist.is_initialized():
            dist.broadcast_object_list([kwargs], src=0)

        # Generate
        try:
            video_tensor = model.generate(**kwargs)
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            if dist.is_initialized(): dist.barrier()
            return {"error": str(e)}

        if dist.is_initialized():
            dist.barrier()

        # Save to temp file and stream back
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        save_video(
            tensor=video_tensor[None],
            save_file=out_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        # Merge audio if s2v
        if "s2v" in args_config.task and audio_path:
            merge_video_audio(out_file, audio_path)

        # Cleanup temp files
        if audio_path: os.remove(audio_path)
        # if s2v image temp file...

        return FileResponse(out_file, media_type="video/mp4", filename="generated.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model loading args (subset of generate.py)
    parser.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--dit_fsdp", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--use_prompt_extend", action="store_true")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen")
    parser.add_argument("--prompt_extend_model", type=str, default=None)
    parser.add_argument("--prompt_extend_target_lang", type=str, default="zh")
    parser.add_argument("--use_relighting_lora", action="store_true")
    
    # Server args
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    
    load_model(args)
    
    if rank == 0:
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # Should not reach here as worker_loop is infinite
        pass