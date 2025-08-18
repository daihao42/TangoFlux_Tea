from diffusers import AutoencoderOobleck
import torch
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers import FluxTransformer2DModel
from torch import nn
from typing import List
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import copy
import torch.nn.functional as F
import numpy as np
from tangoflux.model import TangoFlux
from huggingface_hub import snapshot_download
from tqdm import tqdm
from typing import Optional, Union, List
from datasets import load_dataset, Audio
from math import pi
import json
import inspect
import yaml
from safetensors.torch import load_file


class TangoFluxInference:

    def __init__(
        self,
        name="declare-lab/TangoFlux",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):

        self.vae = AutoencoderOobleck()

        paths = snapshot_download(repo_id=name)
        vae_weights = load_file("{}/vae.safetensors".format(paths))
        self.vae.load_state_dict(vae_weights)
        weights = load_file("{}/tangoflux.safetensors".format(paths))

        with open("{}/config.json".format(paths), "r") as f:
            config = json.load(f)
        self.model = TangoFlux(config)
        self.model.load_state_dict(weights, strict=False)
        # _IncompatibleKeys(missing_keys=['text_encoder.encoder.embed_tokens.weight'], unexpected_keys=[]) this behaviour is expected
        self.vae.to(device)
        self.model.to(device)

    def generate(self, prompt, steps=25, duration=10, guidance_scale=4.5):

        with torch.no_grad():
            latents = self.model.inference_flow(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            wave = self.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * self.vae.config.sampling_rate)
        wave = wave[:, :waveform_end]
        return wave

    def generate_latents(self, prompt, steps=25, duration=10, guidance_scale=4.5, num_samples_per_prompt=1, latent_length=215):

        x0 = torch.randn(num_samples_per_prompt, 645, 64).cuda()


        with torch.no_grad():
            latents = self.model.inference_flow(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                init_latents=x0,
                num_samples_per_prompt=num_samples_per_prompt,
            )

        x1 = latents.transpose(2, 1)
        x0 = x0.transpose(2, 1)

        return x0[:,:, :latent_length], x1[:,:, :latent_length]


    def generate_latents_batch(self, prompt, steps=25, duration=10, guidance_scale=4.5, num_samples_per_prompt=1, latent_length=215):

        if isinstance(prompt, str):
            prompt = [prompt]
        x0 = torch.randn(num_samples_per_prompt * len(prompt), 645, 64)

        with torch.no_grad():
            latents = self.model.inference_flow_batch(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                init_latents=x0,
                num_samples_per_prompt=num_samples_per_prompt,
            )

        x1 = latents.transpose(2, 1)
        x0 = x0.transpose(2, 1)

        return x0[:,:, :latent_length], x1[:,:, :latent_length]
