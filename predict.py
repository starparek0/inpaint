import os
import tempfile
import requests
import torch
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline
from cog import BasePredictor, Input, Path

def download_lora(url: str) -> dict:
    """
    Pobiera plik wag LoRA z zadanego URL i ładuje go jako state_dict.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Nie udało się pobrać wag LoRA spod adresu {url}")
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        state_dict = torch.load(tmp_path, map_location="cuda")
    finally:
        os.remove(tmp_path)
    return state_dict

def apply_lora(unet, lora_state_dict: dict, alpha: float = 1.0):
    """
    Nakłada modyfikacje wag z LoRA na model UNet.
    Uwaga: Ten przykład zakłada, że klucze w state_dict LoRA odpowiadają kluczom w modelu UNet.
    """
    unet_state = unet.state_dict()
    for key, value in lora_state_dict.items():
        if key in unet_state:
            unet_state[key] = unet_state[key] + alpha * value
        else:
            print(f"Klucz {key} nie został znaleziony w UNet")
    unet.load_state_dict(unet_state)

class Predictor(BasePredictor):
    def setup(self):
        """Ładuje bazowy model inpaintingu (flux‑dev‑inpainting)."""
        model_id = "zsxkib/flux-dev-inpainting"  # Upewnij się, że ten model jest dostępny
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()  # Opcjonalnie, dla mniejszego zużycia pamięci

    def predict(
        self,
        prompt: str = Input(
            description="Opis promptu",
            default="A beautiful landscape with mountains"
        ),
        init_image: Path = Input(
            description="Obraz bazowy (plik PNG lub JPG)"
        ),
        mask_image: Path = Input(
            description="Maska (obszar do modyfikacji – biały: generuj, czarny: zachowaj)"
        ),
        lora_model: str = Input(
            description="URL (lub link do Hugging Face) do pliku wag LoRA (.pt)",
            default="https://huggingface.co/your-username/your-lora-model/resolve/main/lora.pt"
        ),
        lora_alpha: float = Input(
            description="Siła modyfikacji LoRA (alpha)",
            default=1.0,
            ge=0.0,
            le=2.0,
        ),
        num_inference_steps: int = Input(
            description="Liczba kroków inferencji",
            default=50,
            ge=1,
            le=150,
        ),
        guidance_scale: float = Input(
            description="Skala guidance",
            default=7.5,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="Seed generatora",
            default=42
        ),
    ) -> Path:
        # Ustawienie seeda dla deterministyczności
        generator = torch.Generator("cuda").manual_seed(seed)

        # Załaduj obraz bazowy oraz maskę
        init_image_pil = Image.open(init_image).convert("RGB")
        mask_image_pil = Image.open(mask_image).convert("RGB")

        # Pobierz i załaduj wagi LoRA
        self.print("Pobieram wagi LoRA...")
        lora_state_dict = download_lora(lora_model)
        self.print("Nakładam wagi LoRA na model UNet...")
        apply_lora(self.pipe.unet, lora_state_dict, alpha=lora_alpha)

        # Generowanie obrazu – obszary wskazane maską zostaną zmodyfikowane przez LoRA
        self.print("Generuję obraz...")
        output = self.pipe(
            prompt=prompt,
            image=init_image_pil,
            mask_image=mask_image_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        result_image = output.images[0]

        # Zapisz wynik do pliku
        output_path = "output.png"
        result_image.save(output_path)
        return Path(output_path)
