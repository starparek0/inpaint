from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List, Iterator
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline
)
from torchvision import transforms
from weights import WeightsDownloadCache
from transformers import CLIPImageProcessor
from lora_loading_patch import load_lora_into_transformer
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)

MAX_IMAGE_SIZE = 1440
MODEL_CACHE = "FLUX.1-dev"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"

def download_weights(url, dest, file=False):
    start = time.time()
    print(f"🔄 Downloading {url} to {dest}...")
    try:
        if not file:
            subprocess.check_call(["pget", "-xf", url, dest], timeout=600)
        else:
            subprocess.check_call(["pget", url, dest], timeout=600)
    except subprocess.TimeoutExpired:
        print("❌ Download timeout exceeded (10 minutes). Retrying...")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        raise
    print(f"✅ Download completed in {time.time() - start:.2f} seconds.")

class Predictor(BasePredictor):
    def setup(self) -> None:
        start = time.time()
        print("🚀 Starting setup()...")
        
        self.weights_cache = WeightsDownloadCache()
        self.last_loaded_lora = None

        try:
            print("🔍 Checking safety model...")
            if not os.path.exists(SAFETY_CACHE):
                download_weights(SAFETY_URL, SAFETY_CACHE)
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                SAFETY_CACHE, torch_dtype=torch.float16
            ).to("cuda")
            print("✅ Safety model loaded.")
        except Exception as e:
            print(f"❌ Safety model loading failed: {e}")
            raise

        try:
            print("🔍 Checking feature extractor...")
            self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
            print("✅ Feature extractor loaded.")
        except Exception as e:
            print(f"❌ Feature extractor loading failed: {e}")
            raise

        try:
            print("🔍 Checking Flux txt2img Pipeline...")
            if not os.path.exists(MODEL_CACHE):
                download_weights(MODEL_URL, '.')
            self.txt2img_pipe = FluxPipeline.from_pretrained(
                MODEL_CACHE,
                torch_dtype=torch.float16,
                cache_dir=MODEL_CACHE
            ).to("cuda")
            print("✅ txt2img model loaded.")
        except Exception as e:
            print(f"❌ txt2img model loading failed: {e}")
            raise
        
        try:
            print("🔍 Initializing img2img pipeline...")
            self.img2img_pipe = FluxImg2ImgPipeline(
                transformer=self.txt2img_pipe.transformer,
                scheduler=self.txt2img_pipe.scheduler,
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
            ).to("cuda")
            print("✅ img2img model loaded.")
        except Exception as e:
            print(f"❌ img2img model loading failed: {e}")
            raise

        try:
            print("🔍 Initializing inpainting pipeline...")
            self.inpaint_pipe = FluxInpaintPipeline.from_pretrained(
                MODEL_CACHE,
                torch_dtype=torch.float16,
                cache_dir=MODEL_CACHE
            ).to("cuda")
            print("✅ Inpainting model loaded.")
        except Exception as e:
            print(f"❌ Inpainting model loading failed: {e}")
            raise

        print(f"✅ Setup completed in {time.time() - start:.2f} seconds.")

    def predict(self, prompt: str, aspect_ratio: str = "1:1", image: Path = None, mask: Path = None, hf_lora: str = None, lora_scale: float = 0.8) -> Path:
        print(f"🚀 Running prediction with prompt: {prompt}")
        width, height = ASPECT_RATIOS[aspect_ratio]
        
        if hf_lora:
            print(f"🔄 Loading LoRA from {hf_lora} with scale {lora_scale}")
            self.txt2img_pipe.load_lora_weights(hf_lora, scale=lora_scale)
            self.img2img_pipe.load_lora_weights(hf_lora, scale=lora_scale)
            self.inpaint_pipe.load_lora_weights(hf_lora, scale=lora_scale)
            print("✅ LoRA loaded successfully.")
        
        if image and mask:
            print("🔍 Running inpainting mode...")
            pipe = self.inpaint_pipe
            init_image = Image.open(image).convert("RGB")
            mask_image = Image.open(mask).convert("RGB")
            output = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        elif image:
            print("🔍 Running img2img mode...")
            pipe = self.img2img_pipe
            init_image = Image.open(image).convert("RGB")
            output = pipe(prompt=prompt, image=init_image).images[0]
        else:
            print("🔍 Running txt2img mode...")
            pipe = self.txt2img_pipe
            output = pipe(prompt=prompt, width=width, height=height).images[0]
        
        output_path = Path("/tmp/output.png")
        output.save(output_path)
        print(f"✅ Image saved at {output_path}")
        return output_path
