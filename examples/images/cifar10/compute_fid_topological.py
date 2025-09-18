# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys

import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.initial_distribution import HeatGP
from torchcfm.fourier_transform import grid_laplacian_eigenpairs
from torchcfm.conditional_flow_matching import *
from torchcfm.fourier_transform import NaiveGridFourierTransform, grid_laplacian_eigenpairs

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")
flags.DEFINE_integer("seed", 0, help="Seed for reproducibility")

flags.DEFINE_string("p0", "gp", help="initial distribution type")
flags.DEFINE_float("c", 0.1, help="c parameter for topological flow matcher")
flags.DEFINE_string("loss", "time_dependent", help="loss type")
flags.DEFINE_string("ft_grid", "3d", help="grid Fourier transform type")
flags.DEFINE_string("boundary_conditions", "neumann", help="boundary conditions")

FLAGS(sys.argv)


def build_eigenbasis(input_shape):
    C, H, W = input_shape
    if FLAGS.ft_grid == "2d":
        eigvecs, eigvals = grid_laplacian_eigenpairs(
            shape=(H, W),
            boundary_conditions=FLAGS.boundary_conditions,
            device=device,
        )
        eigvecs = torch.kron(torch.eye(C, device=device, dtype=eigvecs.dtype), eigvecs)
        eigvals = torch.kron(torch.ones(C, device=device, dtype=eigvals.dtype), eigvals)
        return eigvecs, eigvals
    elif FLAGS.ft_grid == "3d":
        eigvecs, eigvals = grid_laplacian_eigenpairs(
            shape=(C, H, W),
            boundary_conditions=FLAGS.boundary_conditions,
            device=device,
        )
        return eigvecs, eigvals
    raise NotImplementedError(f"Unknown grid Fourier transform type: {FLAGS.ft_grid}")


def build_fm() -> ConditionalTopologicalFlowMatcher:
    sigma = 0.0
    input_shape = (3, 32, 32)
    if FLAGS.model == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "cfm_top":
        c = FLAGS.c
        eigvecs, eigvals = build_eigenbasis(input_shape)
        fourier_transform = NaiveGridFourierTransform(
            shape=input_shape,
            eigenvectors=eigvecs,
        )
        return ConditionalTopologicalFlowMatcher(
            c=c, 
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            fourier_transform=fourier_transform,
        )
    elif FLAGS.model == "otcfm_top":
        c = FLAGS.c
        eigvecs, eigvals = build_eigenbasis(input_shape)
        fourier_transform = NaiveGridFourierTransform(
            shape=input_shape,
            eigenvectors=eigvecs,
        )
        return ExactOptimalTransportConditionalTopologicalFlowMatcher(
            c=c,
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            fourier_transform=fourier_transform,
        )
    raise NotImplementedError(
        f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
    )


def build_p0() -> HeatGP:
    input_shape = (3, 32, 32)
    c = FLAGS.c
    if FLAGS.p0 == "gp":
        eigvecs, eigvals = build_eigenbasis(input_shape)
        return HeatGP(eigvals, eigvecs, c, input_shape, device)
    elif FLAGS.p0 == "normal":
        return torch.distributions.Normal(
            torch.zeros(*input_shape, device=device), torch.ones(*input_shape, device=device)
        )
    else:
        raise NotImplementedError(
            f"Unknown p0 {FLAGS.p0}, must be one of ['gp', 'normal']"
        )


# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Using seed: ", FLAGS.seed)
torch.manual_seed(FLAGS.seed)

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
SAVE_NAME = FLAGS.model + "-" + FLAGS.p0 + "-" + FLAGS.loss + "-" + str(FLAGS.c)
if FLAGS.ft_grid == "2d":
    SAVE_NAME += "-" + FLAGS.ft_grid
if FLAGS.seed != 0:
    SAVE_NAME += "-" + str(FLAGS.seed)
PATH = f"{FLAGS.input_dir}/{SAVE_NAME}/{SAVE_NAME}_cifar10_weights_step_{FLAGS.step}.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()


p0 = build_p0()
FM = build_fm()
vector_field = TopologicalVectorField(u_over_kappa_fn=new_net, fm=FM)


# Define the integration method if euler is used
if FLAGS.integration_method == "euler":
    node = NeuralODE(vector_field, solver=FLAGS.integration_method)


def gen_1_img(unused_latent):
    with torch.no_grad():
        x = p0.sample((FLAGS.batch_size_fid, ))
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                vector_field, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
    traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    return img

for split in ["train", "test"]:
    print(f"Start computing FID on {split} split")
    score = fid.compute_fid(
        gen=gen_1_img,
        dataset_name="cifar10",
        batch_size=FLAGS.batch_size_fid,
        dataset_res=32,
        num_gen=FLAGS.num_gen,
        dataset_split=split,
        mode="legacy_tensorflow",
    )
    print()
    print(f"FID has been computed on {split} split")
    # print()
    # print("Total NFE: ", new_net.nfe)
    print()
    print(f"FID on {split} split: ", score)
