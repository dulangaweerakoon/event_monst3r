import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def _make_grid(B, H, W, device, dtype):
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B, H, W, 2]

def _flow_to_norm(flow):
    # flow: [B, 2, H, W] in pixels -> normalized grid offsets [B, H, W, 2]
    B, _, H, W = flow.shape
    fx = flow[:, 0] * (2.0 / max(W - 1, 1))
    fy = flow[:, 1] * (2.0 / max(H - 1, 1))
    return torch.stack([fx, fy], dim=-1)  # [B, H, W, 2]

@torch.no_grad()
def forward_backward_error(flow_ij, flow_ji):
    """
    flow_ij: [B, 2, H, W]
    flow_ji: [B, 2, H, W]
    Returns:
      err: [B, H, W] forward-backward endpoint error
    """
    B, _, H, W = flow_ij.shape
    device, dtype = flow_ij.device, flow_ij.dtype

    base = _make_grid(B, H, W, device, dtype)            # [B, H, W, 2]
    grid_ij = base + _flow_to_norm(flow_ij)              # sample locations in j

    # sample flow_ji at those locations
    flow_ji_warp = F.grid_sample(
        flow_ji, grid_ij, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [B, 2, H, W]

    # consistency residual
    resid = flow_ij + flow_ji_warp                        # [B, 2, H, W]
    err = torch.norm(resid, dim=1)                        # [B, H, W]
    return err

def _make_base_grid(B: int, H: int, W: int, device, dtype):
    """
    Returns a base sampling grid for grid_sample with shape [B, H, W, 2]
    in normalized coordinates [-1, 1].
    """
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
    grid = torch.stack([grid_x, grid_y], dim=-1)           # [H, W, 2]
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)            # [B, H, W, 2]

def _warp_image(img: torch.Tensor, flow_xy: torch.Tensor) -> torch.Tensor:
    """
    img:     [N, 1, H, W]
    flow_xy: [N, 2, H, W] in pixel units (dx, dy)
    returns: [N, 1, H, W]
    """
    N, _, H, W = img.shape
    device, dtype = img.device, img.dtype

    base = _make_base_grid(N, H, W, device, dtype)  # [N, H, W, 2]

    # Convert pixel flow to normalized flow for grid_sample
    # grid_sample expects x in [-1, 1], so 1 pixel corresponds to 2/(W-1) in x and 2/(H-1) in y.
    fx = flow_xy[:, 0:1] * (2.0 / max(W - 1, 1))  # [N, 1, H, W]
    fy = flow_xy[:, 1:2] * (2.0 / max(H - 1, 1))  # [N, 1, H, W]
    flow_norm = torch.cat([fx, fy], dim=1).permute(0, 2, 3, 1)  # [N, H, W, 2]

    grid = base + flow_norm
    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped

def _voxel_to_warped_image(
    voxel: torch.Tensor,
    flow_xy: torch.Tensor,
    t_bins: torch.Tensor,
    t_ref: float = 0.5,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """
    voxel:  [N, 5, H, W] (can be signed or unsigned, counts or normalized)
    flow_xy:[N, 2, H, W] flow in pixel units
    t_bins: [5] bin centers in [0, 1]
    Returns a single warped accumulation image: [N, 1, H, W]
    """
    N, B, H, W = voxel.shape
    assert B == t_bins.numel() == 5

    # Optional: small blur to stabilize optimization (implemented as avg pool approximation if desired)
    def maybe_blur(x):
        if blur_sigma <= 0:
            return x
        # Simple and cheap blur: 3x3 average pool, repeated a few times
        k = 3
        p = 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=p)

    # Accumulate warped bins to reference time
    acc = torch.zeros((N, 1, H, W), device=voxel.device, dtype=voxel.dtype)
    for b in range(B):
        dt = float(t_bins[b].item() - t_ref)  # scalar
        # Warp bin by flow scaled by dt
        bin_img = voxel[:, b:b+1]  # [N, 1, H, W]
        bin_img = maybe_blur(bin_img)
        warped = _warp_image(bin_img, flow_xy * dt)
        acc = acc + warped

    acc = maybe_blur(acc)
    return acc

def _normalized_correlation(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    a, b: [N, 1, H, W]
    returns: [N] correlation coefficient per batch element
    """
    N = a.shape[0]
    a_flat = a.view(N, -1)
    b_flat = b.view(N, -1)

    a0 = a_flat - a_flat.mean(dim=1, keepdim=True)
    b0 = b_flat - b_flat.mean(dim=1, keepdim=True)

    num = (a0 * b0).mean(dim=1)
    den = (a0.pow(2).mean(dim=1).sqrt() * b0.pow(2).mean(dim=1).sqrt()).clamp_min(eps)
    return num / den

class EventContrastFlow(torch.nn.Module):
    """
    Contrast maximization based event flow estimator.

    Inputs:
      voxel_a: [N, 5, H, W]
      voxel_b: [N, 5, H, W]

    Output:
      flow:    [N, 2, H, W] in pixel units
    """
    def __init__(
        self,
        coarse_grid: tuple[int, int] = (32, 32),
        iters: int = 100,
        lr: float = 0.01,
        t_ref: float = 0.5,
        lambda_var: float = 0.1,
        blur_sigma: float = 0.0,
        flow_clip: float = 50.0,
        device: str | None = None,
    ):
        super().__init__()
        self.coarse_grid = coarse_grid
        self.iters = iters
        self.lr = lr
        self.t_ref = t_ref
        self.lambda_var = lambda_var
        self.blur_sigma = blur_sigma
        self.flow_clip = flow_clip
        self.device_override = device

        # Fixed 5 bin centers, assuming uniform bins across the interval
        # You can change these if your voxel bins correspond to a different temporal layout.
        self.register_buffer("t_bins", torch.linspace(0.0, 1.0, 5))

    def forward(self, voxel_a: torch.Tensor, voxel_b: torch.Tensor) -> torch.Tensor:
        if self.device_override is not None:
            voxel_a = voxel_a.to(self.device_override)
            voxel_b = voxel_b.to(self.device_override)

        N, B, H, W = voxel_a.shape
        assert B == 5 and voxel_b.shape == (N, 5, H, W)

        Gh, Gw = self.coarse_grid
        device, dtype = voxel_a.device, voxel_a.dtype

        # Optimize a coarse flow grid per pair, then upsample to H, W.
        # flow_coarse: [N, 2, Gh, Gw]
        flow_coarse = torch.zeros((N, 2, Gh, Gw), device=device, dtype=dtype, requires_grad=True)

        opt = torch.optim.Adam([flow_coarse], lr=self.lr)

        for _ in range(self.iters):
            opt.zero_grad(set_to_none=True)

            # Upsample coarse flow to full res
            flow_full = F.interpolate(flow_coarse, size=(H, W), mode="bilinear", align_corners=True)
            # print(flow_coarse.requires_grad,flow_full.requires_grad)

            # Optional: clip to avoid exploding updates in early iterations
            if self.flow_clip is not None and self.flow_clip > 0:
                flow_full = flow_full.clamp(-self.flow_clip, self.flow_clip)

            # Warp and accumulate each voxel into a single image using the same flow hypothesis
            Ia = _voxel_to_warped_image(
                voxel_a, flow_full, self.t_bins, t_ref=self.t_ref, blur_sigma=self.blur_sigma
            )
            Ib = _voxel_to_warped_image(
                voxel_b, flow_full, self.t_bins, t_ref=self.t_ref, blur_sigma=self.blur_sigma
            )

            # Objective:
            # 1) maximize correlation between warped images from voxel_a and voxel_b
            # 2) encourage each warped image to be sharp (high variance)
            corr = _normalized_correlation(Ia, Ib).mean()

            var_a = Ia.view(N, -1).var(dim=1, unbiased=False).mean()
            var_b = Ib.view(N, -1).var(dim=1, unbiased=False).mean()

            score = corr + self.lambda_var * (var_a + var_b)

            # We minimize negative score
            loss = -score
            loss.backward()
            opt.step()

        # Final dense flow
        flow_full = F.interpolate(flow_coarse.detach(), size=(H, W), mode="bilinear", align_corners=True)
        if self.flow_clip is not None and self.flow_clip > 0:
            flow_full = flow_full.clamp(-self.flow_clip, self.flow_clip)
        return flow_full
    



def flow_to_rgb(flow: torch.Tensor, max_mag: float | None = None) -> np.ndarray:
    """
    flow: [2, H, W] torch tensor (dx, dy) in pixel units
    returns: [H, W, 3] uint8 RGB image
    """
    flow = flow.detach().cpu().numpy()
    fx, fy = flow[0], flow[1]

    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)  # [-pi, pi]

    if max_mag is None:
        max_mag = np.percentile(mag, 99) + 1e-6

    mag = np.clip(mag / max_mag, 0, 1)

    # HSV image
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)  # hue [0,1]
    hsv[..., 1] = mag                          # saturation
    hsv[..., 2] = 1.0                          # value

    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb


def estimate_event_flow_in_chunks(
    voxel_pairs_a: torch.Tensor,
    voxel_pairs_b: torch.Tensor,
    estimator: EventContrastFlow,
    chunk_size: int = 12,
) -> torch.Tensor:
    """
    Mimics the RAFT chunking pattern and returns [num_pairs, 2, H, W].

    voxel_pairs_a: [num_pairs, 5, H, W]
    voxel_pairs_b: [num_pairs, 5, H, W]
    """
    num_pairs = voxel_pairs_a.shape[0]
    flows = []
    for i in range(0, num_pairs, chunk_size):
        end = min(i + chunk_size, num_pairs)
        flow = estimator(voxel_pairs_a[i:end], voxel_pairs_b[i:end])  # [B, 2, H, W]
        flows.append(flow)
    return torch.cat(flows, dim=0)