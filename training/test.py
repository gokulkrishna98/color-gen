from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

A_PROMPT_DEFAULT = "best quality, extremely detailed"
N_PROMPT_DEFAULT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

def apply_color(image, color_map):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2LAB)

    l, _, _ = cv2.split(image)
    _, a, b = cv2.split(color_map)

    merged = cv2.merge([l, a, b])

    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return result

def run_sampler(
    model,
    input_image: np.ndarray,
    prompt: str,
    num_samples: int = 1,
    image_resolution: int = 256,
    seed: int = -1,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    guess_mode=False,
    strength=1.0,
    ddim_steps=20,
    eta=0.0,
    scale=9.0,
    show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        ddim_sampler = DDIMSampler(model)

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            show_progress=show_progress,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
        colored_results = [apply_color(img, result) for result in results]

        return colored_results

apply_canny = CannyDetector()
checkpoint_path = './models/model_weights_0.pth'
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(checkpoint_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

image = cv2.imread('./training/images_black/n03888257_8502.JPEG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if necessary
image = image.astype(np.uint8)  # Normalize the image to [0, 1]

results = run_sampler(
    model = model,
    input_image = image,
    prompt = "",
    image_resolution = 512
)

cv2.imwrite('./training/output.JPEG', results[0])