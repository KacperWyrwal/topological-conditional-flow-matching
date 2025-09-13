import torch 


def heat_gp_sample(sample_shape: tuple[int, ...], input_shape: tuple[int, ...], eigvals: torch.Tensor, eigvecs: torch.Tensor, c: float, device: torch.device) -> torch.Tensor:
    eps = torch.randn(sample_shape + eigvals.shape, device=device)
    spectral_density = torch.exp(-eigvals * c).sqrt()
    res = torch.einsum("kd, ...k -> ...d", eigvecs, eps * spectral_density)
    res = torch.reshape(res, sample_shape + input_shape)
    return res


class HeatGP:
    def __init__(self, eigvals: torch.Tensor, eigvecs: torch.Tensor, c: float, input_shape: tuple[int, ...], device: torch.device):
        self.eigvals = eigvals
        self.eigvecs = eigvecs.mT
        self.c = c
        self.input_shape = input_shape
        self.device = device

    def sample(self, shape: tuple[int, ...] = torch.Size([])) -> torch.Tensor:
        return heat_gp_sample(shape, self.input_shape, self.eigvals, self.eigvecs, self.c, self.device)


def build_p0(name: str = 'normal', eigvals: torch.Tensor = None, eigvecs: torch.Tensor = None, c: float = None, input_shape: tuple[int, ...] = None, device: torch.device = None) -> HeatGP | torch.distributions.Distribution:
    if name == 'gp':
        return HeatGP(eigvals, eigvecs, c, input_shape, device)
    elif name == 'normal':
        normal = torch.distributions.Normal(torch.zeros(*input_shape), torch.ones(*input_shape))
        return normal
    else:
        raise ValueError(f"Invalid name: {name}")