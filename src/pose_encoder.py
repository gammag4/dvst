import einx

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pad(hw, p):
    # Pads the input so that it is divisible by 'p'
    # hw: (2,), p: (1)
    
    pad_raw = [((p - i) % p) for i in hw]
    pad_s = [i // 2 for i in pad_raw]
    pad = (pad_s[1], pad_raw[1] - pad_s[1], pad_s[0], pad_raw[0] - pad_s[0])
    hw_padded = [i + d for i, d in zip(hw, pad_raw)]
    
    # pad: (pad_width_start, pad_width_end, pad_height_start, pad_height_end) (starts from last dimension to pad)
    # hw_padded: (2,), pad: (4,)
    return hw_padded, pad


def compute_view_rays(vecs, Kinv, R, t):
    # Computes view rays (o, d)
    # vecs: meshgrid vecs, first dim is (x, y, z)
    # vecs: (3, h, w), Kinv: (3, 3), R: (B, 3, 3), t: (B, 3)

    # TODO check without double precision
    vecs, Kinv, R, t = [i.to(torch.float64) for i in (vecs, Kinv, R, t)]

    h, w = vecs.shape[-2:]

    o = -einx.dot('... h w, ... h -> ... w', R, t)  # -R^T t
    o = einx.rearrange('... c -> ... c h w', o, h=h, w=w) # repeat o for each vec # TODO repeating maybe not needed
    d = einx.dot('... x1 c2, x1 c, c h w -> ... c2 h w', R.to(torch.float64), Kinv.to(torch.float64), vecs) # R^T K^-1 x_ij,cam # TODO check without double precision
    d = d / einx.sum('b [c] h w -> b 3 h w', d * d).sqrt() # normalize d

    # o, d: (B, 3, H, W)
    return o, d


def compute_plucker_rays(o, d):
    # o, d: (B, 3, H, W)

    l = torch.cross(o, d, dim=-3)
    rays = torch.concat([d, l], dim=-3)

    # rays: (B, 6, H, W)
    return rays


def compute_octaves(v, n_oct, dim=-1):
    assert dim < 0, 'No positive dim allowed'

    v = v * torch.pi
    tensors = [torch.sin(v), torch.cos(v)]
    last = v
    for _ in range(n_oct - 1):
        last = last * 2
        tensors.append(torch.sin(last))
        tensors.append(torch.cos(last))

    return torch.stack(tensors, dim=dim).flatten(dim - 1, dim)


class PoseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_lat = self.config.model.d_lat
        self.n_oct = self.config.model.n_oct
        self.C = self.config.model.C
        self.p = self.config.model.p

        # TODO test two cases, one with parameter (this) and another with two different linear layers one for sources (w/ images) and another for target (w/o images)
        # TODO initialize w gaussian
        # (C, p, p)
        self.im_parameter = nn.Parameter(torch.zeros((self.C, self.p, self.p)))

        # TODO check without double precision
        self.linear = nn.Linear(
            in_features=(12 * self.n_oct + self.C) * self.p ** 2 + 2 * self.n_oct,
            #in_features=(6 + self.C) * self.p ** 2 + 1, # Without octaves, just for testing
            out_features=self.d_lat,
            dtype=torch.float64
        )
        
    def _compute_view_rays(self, Kinv, R, t, pad, hw):
        # The forward function was split into two to display the view rays layer
        
        pad_s = pad[-2::-2]

        # Creates vectors for each pixel in screen
        # No need to unflip y axis since it being flipped does not affect the topological structure of the representation TODO is it true?
        ranges = [torch.arange(l, dtype=torch.float64) - o + 0.5 for o, l in zip(pad_s, hw)]
        # In the original LVSM impl, the K^{-1} multiplication is done here bc its faster, maybe change the code to do that too (https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/utils/data_utils.py#L71-L73)
        # Used torch.ones since it seems to be used by most of the vision models similar to this (e.g. lvsm, see https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/utils/data_utils.py#L73)
        # The torch.ones is used bc the convention is that the theoretical sensor plane has focal length 1 (it maps to coordinates (u, v, 1), which would be equivalent to (f u, f v, f) = f(u, v, 1))
        vecs = torch.meshgrid(*ranges, indexing='ij')
        vecs = torch.concat([torch.stack([*vecs[::-1]]), torch.ones((1, *vecs[0].shape))], dim=-3)

        o, d = compute_view_rays(vecs, Kinv, R, t) # o, d: (B, 3, H, W)
        return o, d

    # I = images, HW = tuple with height and width
    # Set both if image has been resized, specifying original image height and width in HW
    # We assume images are already resized (always resize them maintaining aspect ratio)
    # We assume images are already padded so that p divides H and W
    # We assume that the K matrix uses xy mapping instead of uv (sensor area is real in range [(0, 0), (h, w)], not [(0, 0), (1, 1)])
    # We assume images are in type float with colors in range 0-1
    def forward(self, Kinv, R, t, time, I=None, hw=None):
        # I: (B, C, H, W), Kinv: (3, 3), R: (B, 3, 3), t: (B, 3), time: (B,), hw: (2,)
 
        #TODO corrige hw
        #TODO tem que retornar quanto de padding teve pra tirar o padding na comparacao da loss function
        #TODO na verdade no lugar de retornar o padding ja retorna a visao prevista com padding retirado no modelo final
        
        assert (I == None) ^ (hw == None), 'Either I or HW or both should be set'
        
        if I is not None:
            hw = I.shape[-2:]
            I = I * 2 - 1 # Normalizing image

        # Pads the input so that it is divisible by 'p'
        hw, pad = compute_pad(hw, self.p)
        I = F.pad(I, pad, 'constant', 0) if I is not None else None

        o, d = self._compute_view_rays(Kinv, R, t, pad, hw)
        plucker_rays = compute_plucker_rays(o, d) # (B, 6, H, W)

        # (B, 2 * 6 * n_oct, H, W)
        plucker_octs = compute_octaves(plucker_rays, self.n_oct, dim=-3)
        #plucker_octs = torch.concat([plucker_octs, I * 2 - 1], dim=-3) if I is not None else plucker_octs # Transforming and concatenating image

        # Concatenating image with octaves and rearranging into patches
        # (B, HW/p^2, (12 * n_oct + C) * p^2)
        if I is None:
            patches = einx.rearrange('... c1 (h p1) (w p2), c2 p1 p2 -> ... (h w) ((c1 + c2) p1 p2)', plucker_octs, self.im_parameter, p1=self.p, p2=self.p)
        else:
            patches = einx.rearrange('... c1 (h p1) (w p2), ... c2 (h p1) (w p2) -> ... (h w) ((c1 + c2) p1 p2)', plucker_octs, I, p1=self.p, p2=self.p)

        time_octs = compute_octaves(time.unsqueeze(-1), self.n_oct, dim=-1) # (B, 2 * n_oct)

        # (B, HW/p^2, (12 * n_oct + C) * p^2 + 2 * n_oct)
        embeds = einx.rearrange('... hw c1, ... c2 -> ... hw (c1 + c2)', patches, time_octs)
        embeds = self.linear(embeds)

        return embeds, pad
