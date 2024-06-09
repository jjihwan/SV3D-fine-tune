from typing import Dict, Union

import torch
import torch.nn as nn

from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization
from einops import repeat, rearrange


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor, # [BF,4,72,72]
        sigma: torch.Tensor,
        cond: Dict, #concat:[BF,4,72,72], crossattn:[BF,1,1024], vector:[BF,1280]
        **additional_model_inputs,
    ) -> torch.Tensor:
        # print()
        # print("[denoiser forward]")
        # print(input.shape, input.mean(), input.std())
        # for k, v in cond.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape, v.mean(), v.std())

        # print("sigma")
        # print(sigma.shape, sigma)

        # print("[denoiser forward], uc")
        # print(input[:21].shape, input[:21].mean(), input[:21].std())
        # print(sigma[:21].shape, sigma[:21].mean(), sigma[:21].std())
        # for k, v in cond.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v[:21].shape, v[:21].mean(), v[:21].std())
        
        # print("[denoiser forward], c")
        # print("input", input[21:].shape, input[21:].mean(), input[21:].std())
        # print("cond")
        # for k, v in cond.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v[21:].shape, v[21:].mean(), v[21:].std())
        # print("sigma", sigma[21:].shape, sigma[21:].mean(), sigma[21:].std())

        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        # print()
        # print("network input")
        # print(input.shape, c_in.shape, c_noise.shape, c_out.shape, c_skip.shape)

        return (
            network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
            + input * c_skip
        )
    

class SV3DDenoiser(Denoiser):
    def __init__(self, scaling_config: Dict):
        super().__init__(scaling_config)

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor, # [B,F,C,72,72]
        sigma: torch.Tensor,
        cond: Dict, #concat:[B,4,72,72], crossattn:[B,1,1024], vector:[B,1280]
        **additional_model_inputs,
    ) -> torch.Tensor:
        # print()

        b, f = input.shape[:2]
        input = rearrange(input, "b f ... -> (b f) ...")

        for k in ["crossattn", "concat"]:
            cond[k] = repeat(cond[k], "b ... -> b f ...", f=f)
            cond[k] = rearrange(cond[k], "b f ... -> (b f) ...", f=f)

        additional_model_inputs["image_only_indicator"] = torch.zeros((b,f)).to(input.device, input.dtype)
        additional_model_inputs["num_video_frames"] = f

        # print("[SV3D denoiser forward]")
        # print(input.shape, input.mean(), input.std())
        # for k, v in cond.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape, v.mean(), v.std())
        # print("sigma", sigma.shape, sigma)

        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        
        network_output = network(input * c_in, c_noise, cond, **additional_model_inputs)
        # print(network_output.shape, input.shape)
        return network_output * c_out + input * c_skip
    
        return (
            network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
            + input * c_skip
        )


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(
            discretization_config
        )
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
