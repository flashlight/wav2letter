#!/usr/bin/env python3

import argparse

import torch
from wav2letter.criterion import ASGLoss, CriterionScaleMode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu", action="store_true", help="Use cpu backend, otherwise use CUDA backend"
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="store tensors in double, otherwise in float",
    )
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")
    float_type = torch.double if args.double else torch.float

    # create ASG loss with scaling the loss to the sqrt of target size
    # and 6 tokens (6 tokens scores predicted by some network for each frame)
    asg = ASGLoss(6, scale_mode=CriterionScaleMode.TARGET_SZ_SQRT).to(device)
    # define the input to the loss (scores for tokens came from
    # some network for each frame) size is [batch, time, ntokens]
    input = torch.tensor(
        [
            [
                [-0.4340, -0.0254, 0.3667, 0.4180, -0.3805, -0.1707],
                [0.1060, 0.3631, -0.1122, -0.3825, -0.0031, -0.3801],
                [0.0443, -0.3795, 0.3194, -0.3130, 0.0094, 0.1560],
                [0.1252, 0.2877, 0.1997, -0.4554, 0.2774, -0.2526],
                [-0.4001, -0.2402, 0.1295, 0.0172, 0.1805, -0.3299],
            ],
            [
                [0.3298, -0.2259, -0.0959, 0.4909, 0.2996, -0.2543],
                [-0.2863, 0.3239, -0.3988, 0.0732, -0.2107, -0.4739],
                [-0.0906, 0.0480, -0.1301, 0.3975, -0.3317, -0.1967],
                [0.4372, -0.2006, 0.0094, 0.3281, 0.1873, -0.2945],
                [0.2399, 0.0320, -0.3768, -0.2849, -0.2248, 0.3186],
            ],
            [
                [0.0225, -0.3867, -0.1929, -0.2904, -0.4958, -0.2533],
                [0.4001, -0.1517, -0.2799, -0.2915, 0.4198, 0.4506],
                [0.1446, -0.4753, -0.0711, 0.2876, -0.1851, -0.1066],
                [0.2081, -0.1190, -0.3902, -0.1668, 0.1911, -0.2848],
                [-0.3846, 0.1175, 0.1052, 0.2172, -0.0362, 0.3055],
            ],
        ],
        dtype=float_type,
        device=device,
        requires_grad=True,
    )
    # define the padded target transcriptions (encoded with tokens indices),
    # padded index is -1
    target = torch.tensor(
        [[2, 1, 5, 1, 3], [4, 3, 5, -1, -1], [3, 2, 2, 1, -1]],
        dtype=torch.int,
        device=device,
    )
    # define target transcriptions sizes
    target_size = torch.tensor([5, 3, 4], dtype=torch.int, device=device)
    # define gradient of the network
    grad = torch.ones(3, dtype=float_type, device=device)

    print("List of ASG parameters", list(asg.parameters()))
    # run forward pass to compute the ASG loss
    loss = asg.forward(input, target, target_size)
    print("ASG loss =", loss)
    # run backward pass
    loss.backward(grad)
    print("Gradients with respect to the ASG loss input", input.grad)
    print("Gradients with respect to the transition matrix", asg.transitions.grad)
