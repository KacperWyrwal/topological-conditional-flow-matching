# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    ConditionalTopologicalFlowMatcher,
    ExactOptimalTransportConditionalTopologicalFlowMatcher,
    TopologicalVectorField,
)
from torchcfm.fourier_transform import NaiveGridFourierTransform, grid_laplacian_eigenpairs
from torchcfm.initial_distribution import HeatGP
from torchcfm.loss import TimeDependentTopologicalLoss, TimeIndependentTopologicalLoss
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("p0", "gp", help="initial distribution type")
flags.DEFINE_string("loss", "time_dependent", help="loss type")
flags.DEFINE_string("ft_grid", "3d", help="grid Fourier transform type")
flags.DEFINE_string("boundary_conditions", "neumann", help="boundary conditions")
flags.DEFINE_integer("seed", 0, help="Seed for reproducibility")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_float("c", 1.0, help="c parameter for topological flow matcher")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


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


def build_loss(fm: ConditionalTopologicalFlowMatcher) -> callable:
    if FLAGS.loss == 'time_dependent':
        return TimeDependentTopologicalLoss(fm)
    elif FLAGS.loss == 'time_independent':
        return TimeIndependentTopologicalLoss(fm)
    else:
        raise ValueError(f"Invalid name: {FLAGS.loss}")


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


def train(argv):
    torch.manual_seed(FLAGS.seed)
    SAVE_NAME = FLAGS.model + "-" + FLAGS.p0 + "-" + FLAGS.loss + "-" + str(FLAGS.c) 
    if FLAGS.ft_grid == "2d":
        SAVE_NAME += "-" + FLAGS.ft_grid
    if FLAGS.seed != 0:
        SAVE_NAME += "-" + str(FLAGS.seed)
    print(
        "lr, total_steps, ema decay, save_step, seed:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
        FLAGS.seed,
    )
    print(f"model, p0, loss, c, ft_grid, boundary_conditions: {FLAGS.model}, {FLAGS.p0}, {FLAGS.loss}, {FLAGS.c}, {FLAGS.ft_grid}, {FLAGS.boundary_conditions}")

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    input_shape = (3, 32, 32)
    net_model = UNetModelWrapper(
        dim=input_shape,
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    ### Flow matcher 
    FM = build_fm()
    p0 = build_p0()
    loss_fn = build_loss(FM)
    net_vector_field = TopologicalVectorField(u_over_kappa_fn=net_model, fm=FM)
    ema_vector_field = TopologicalVectorField(u_over_kappa_fn=ema_model, fm=FM)

    savedir = FLAGS.output_dir + SAVE_NAME + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = p0.sample((FLAGS.batch_size, ))
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = loss_fn(vt, ut, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                #generate_samples(net_vector_field, p0, FLAGS.parallel, savedir, step, net_="normal")
                #generate_samples(ema_vector_field, p0, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{SAVE_NAME}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
