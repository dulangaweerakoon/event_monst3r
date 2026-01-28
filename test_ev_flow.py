import torch
import torch.nn.functional as F
import numpy as np

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from third_party.raft import load_RAFT

class OccMask(torch.nn.Module):
    def __init__(self, th=3):
        super(OccMask, self).__init__()
        self.th = th
        self.base_coord = None

    def init_grid(self, shape, device):
        H, W = shape
        hh, ww = torch.meshgrid(torch.arange(
            H).float(), torch.arange(W).float())
        coord = torch.zeros([1, H, W, 2])
        coord[0, ..., 0] = ww
        coord[0, ..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    @torch.no_grad()
    def get_oob_mask(self, base_coord, flow_1_2):
        target_range = base_coord + flow_1_2.permute([0, 2, 3, 1])
        oob_mask = (target_range[..., 0] < 0) | (target_range[..., 0] > self.W-1) | (
            target_range[..., 1] < 0) | (target_range[..., 1] > self.H-1)
        return ~oob_mask[:, None, ...]

    @torch.no_grad()
    def get_flow_inconsistency_tensor(self, base_coord, flow_1_2, flow_2_1):
        B, C, H, W = flow_1_2.shape
        sample_grids = base_coord + flow_1_2.permute([0, 2, 3, 1])
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        sampled_flow = F.grid_sample(
            flow_2_1, sample_grids, align_corners=True)
        return torch.abs((sampled_flow+flow_1_2).sum(1, keepdim=True))

    def forward(self, flow_1_2, flow_2_1):
        B, _, H, W = flow_1_2.shape
        if self.base_coord is None:
            self.init_grid([H, W], device=flow_1_2.device)
        base_coord = self.base_coord.expand([B, -1, -1, -1])
        oob_mask = self.get_oob_mask(base_coord, flow_1_2)
        flow_inconsistency_tensor = self.get_flow_inconsistency_tensor(
            base_coord, flow_1_2, flow_2_1)
        valid_flow_mask = flow_inconsistency_tensor < self.th
        return valid_flow_mask*oob_mask


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
        # 3, 1
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
        lr: float = 0.2,
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

        for idx in range(self.iters):
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

            cv2.imwrite("tmp/Ia_"+str(idx)+".png", (Ia[0,0].detach().cpu().numpy() / Ia[0,0].max().item() * 255).astype(np.uint8))
            cv2.imwrite("tmp/Ib_"+str(idx)+".png", (Ib[0,0].detach().cpu().numpy() / Ib[0,0].max().item() * 255).astype(np.uint8))

            # Objective:
            # 1) maximize correlation between warped images from voxel_a and voxel_b
            # 2) encourage each warped image to be sharp (high variance)
            corr = _normalized_correlation(Ia, Ib).mean()



            var_a = Ia.view(N, -1).var(dim=1, unbiased=False).mean()
            var_b = Ib.view(N, -1).var(dim=1, unbiased=False).mean()

            

            score = self.lambda_var * (var_a + var_b) + corr

            # We minimize negative score
            loss = -score
            print(f"Iter {idx+1}/{self.iters}: Loss={loss.item():.6f}, Corr={corr.item():.6f}, VarA={var_a.item():.6f}, VarB={var_b.item():.6f}")
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


device = "cuda"

# Example: one pair
# H, W = 384, 512
# voxel_a = torch.randn(1, 5, H, W, device=device)  # your event voxel for image i
# voxel_b = torch.randn(1, 5, H, W, device=device)  # your event voxel for image j

voxel_a = np.load("/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/ani2/event_voxels/voxel_00000.npy")
voxel_b = np.load("/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/ani2/event_voxels/voxel_00010.npy")

voxel_a = torch.from_numpy(voxel_a).unsqueeze(0).to(device)
voxel_b = torch.from_numpy(voxel_b).unsqueeze(0).to(device)

print(voxel_a.requires_grad, voxel_b.requires_grad)

img_a = cv2.imread("/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/ani2/rgbs/rgb_00000.jpg")
img_b = cv2.imread("/storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/ani2/rgbs/rgb_00010.jpg")

img_a = torch.from_numpy(img_a).permute(2,0,1).unsqueeze(0).float().to(device) #/ 255.0
img_b = torch.from_numpy(img_b).permute(2,0,1).unsqueeze(0).float().to(device) #/ 255.0
H, W = voxel_a.shape[2], voxel_a.shape[3]


estimator = EventContrastFlow(
    coarse_grid=(256, 256),
    iters=100,
    lr=0.1,
    lambda_var=0.1,
    blur_sigma=1.0,
    flow_clip=60,
).to(device)

flow_ij = estimator(voxel_a, voxel_b)  # [1, 2, H, W]
flow_ji = estimator(voxel_b, voxel_a)  # [1, 2, H, W] if you want backward flow


flow = flow_ij[0]  # [2, H, W]
ev_flow_rgb = flow_to_rgb(flow)



flow_net = load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
flow_net = flow_net.to(device)
flow_net.eval()

with torch.no_grad():
    # img_a_gray = 0.2989 * img_a[:,0:1] + 0.5870 * img_a[:,1:2] + 0.1140 * img_a[:,2:3]
    # img_b_gray = 0.2989 * img_b[:,0:1] + 0.5870 * img_b[:,1:2] + 0.1140 * img_b[:,2:3]
    # flow_net.initialize(img_a_gray, img_b_gray)
    # flow_net.iters = 40
    # flow_net.flow_clip = 60.0
    # flow_net.coarse_grid = (32, 32)
    # flow_net.lambda_var = 0.05

    flow_ij_nn = flow_net(img_a, img_b, iters=20, test_mode=True)[1]  # [1, 2, H, W] 
    flow_ji_nn = flow_net(img_b, img_a, iters=20, test_mode=True)[1]  # [1, 2, H, W]

print(flow_ij_nn.shape) 
flow_nn = flow_ij_nn[0]  # [2, H, W]
nn_flow_rgb = flow_to_rgb(flow_nn)


# flow_diff = ev_flow_rgb - nn_flow_rgb
# stitch ev flow and nn flow in horizontal direction


# cv2.imwrite("event_flow.png", ev_flow_rgb)


err = forward_backward_error(flow_ij, flow_ji)
mean_err = err.mean().item()
p95_err = err.flatten().kthvalue(int(err.numel() * 0.95))[0].item()
print("Event mean error:", mean_err)


err_nn = forward_backward_error(flow_ij_nn, flow_ji_nn)
mean_err_nn = err_nn.mean().item()
p95_err_nn = err_nn.flatten().kthvalue(int(err_nn.numel() * 0.95))[0].item()
print("NN mean error:", mean_err_nn)



cv2.imwrite("event_flow.png", ev_flow_rgb)
cv2.imwrite("nn_flow.png", nn_flow_rgb)


err0 = err[0].detach().cpu().numpy()
plt.imshow(err0)
plt.colorbar()
plt.title("Forward-backward flow error (pixels)")
plt.axis("off")
plt.savefig("event_flow_error.png")


err1 = err_nn[0].detach().cpu().numpy()
plt.imshow(err1)
plt.colorbar()
plt.title("NN Forward-backward flow error (pixels)")
plt.axis("off")
plt.savefig("nn_flow_error.png")



# get valid flow mask

get_valid_flow_mask = OccMask(th=3.0)
valid_mask = get_valid_flow_mask(flow_ij, flow_ji)  # [1, 1, H, W]
valid_mask_nn = get_valid_flow_mask(flow_ij_nn, flow_ji_nn)  # [1, 1, H, W]

valid_mask_np = valid_mask[0,0].detach().cpu().numpy()
# imwrite as grayscale
cv2.imwrite("event_flow_valid_mask.png", (valid_mask_np * 255).astype(np.uint8))


valid_mask_nn_np = valid_mask_nn[0,0].detach().cpu().numpy()
cv2.imwrite("nn_flow_valid_mask.png", (valid_mask_nn_np * 255).astype(np.uint8))


# get flow diff in valid regions
flow_diff = flow_ij - flow_ij_nn  # [1, 2, H, W]
flow_diff_mag = torch.norm(flow_diff, dim=1, keepdim=True)  # [1, 1, H, W]

cv2.imwrite("flow_diff_mag.png", (flow_diff_mag[0,0].detach().cpu().numpy() / flow_diff_mag.max().item() * 255).astype(np.uint8))
flow_diff_mag_valid = flow_diff_mag * valid_mask_nn  # [1, 1, H, W]
mean_flow_diff_valid = flow_diff_mag_valid.sum().item() / (valid_mask.sum().item() + 1e-6)
print("Mean flow difference in valid regions:", mean_flow_diff_valid)
