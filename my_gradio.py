from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_canny = CannyDetector()

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
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    num_samples: int = 1,
    image_resolution: int = 256,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=9.0,
    seed: int = -1,
    eta=0.0,
    show_progress: bool = True
):
    model = create_model('/Users/ayushnaique28/vs_code/Python/color-gen/models/cldm_v15.yaml').cpu()
    model.load_state_dict(torch.load('/Users/ayushnaique28/vs_code/Python/color-gen/models/gokul.ckpt', map_location="cuda"), strict=False)

    # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model = model.to(torch.device("cuda"))
    ddim_sampler = DDIMSampler(model)
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if necessary
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
        # L = lab[:,:,0]
        input_image = input_image.astype(np.uint8)  # Normalize the image to [0, 1]

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, 100, 200)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().to(torch.device("cuda")) / 255.0
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
        # x_samples
        # print("HELLOOOOOO\n")
        results = [x_samples[i] for i in range(num_samples)]
        colored_results = [apply_color(img, result) for result in results]

        return colored_results + [255 - detected_map] 

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## TEST GRADIO")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", rows=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=run_sampler, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
