#!/usr/bin/env python3

import struct
import sys

import torch
import torch.nn as nn
import wav2letter._criterion as _C


def get_data_ptr_as_bytes(tensor):
    return struct.pack("P", tensor.data_ptr())


def get_cuda_stream_as_bytes():
    s = torch.cuda.current_stream().cuda_stream
    return s.to_bytes(_C.sizeof_cuda_stream, byteorder=sys.byteorder)


def check_tensor(tensor, size, dtype, device):
    shape = torch.Size(size)
    if tensor.shape != shape:
        raise ValueError(f"wrong tensor size: expected {shape}, got {tensor.shape}")
    return tensor.to(dtype=dtype, device=device).contiguous()


def run_direction(cls, device, direction, *args):
    """
    Select and run CPU/CUDA implementation of `forward()` or `backward()`.
    If CUDA, create the right device context and also pass the CUDA stream.
    """
    device = torch.device(device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            fn = getattr(cls.cuda_impl(), direction)
            fn(*args, get_cuda_stream_as_bytes())
    elif device.type == "cpu":
        fn = getattr(cls.cpu_impl(), direction)
        fn(*args)
    else:
        raise ValueError("unknown/unsupported device type")


def run_forward(cls, device, *args):
    run_direction(cls, device, "forward", *args)


def run_backward(cls, device, *args):
    run_direction(cls, device, "backward", *args)


def run_get_workspace_size(cls, device, *args):
    device = torch.device(device)
    if device.type == "cuda":
        return cls.cuda_impl().get_workspace_size(*args)
    elif device.type == "cpu":
        return cls.cpu_impl().get_workspace_size(*args)
    else:
        raise ValueError("unknown/unsupported device type")


def create_workspace(cls, device, *args):
    """
    Select and run CPU/CUDA implementation of `get_workspace_size()`,
    then return a byte tensor of appropriate size.
    """
    workspace_size = run_get_workspace_size(cls, device, *args)
    return torch.empty(workspace_size, dtype=torch.uint8, device=device)


class FACFunction(torch.autograd.Function):
    @staticmethod
    def cuda_impl():
        return _C.CudaForceAlignmentCriterion

    @staticmethod
    def cpu_impl():
        return _C.CpuForceAlignmentCriterion

    @classmethod
    def forward(cls, ctx, input, target, target_size, trans, scale_mode):
        B = input.size(0)
        T = input.size(1)
        N = input.size(2)
        L = target.size(1)
        device = input.device

        input_float = check_tensor(input, [B, T, N], torch.float, device)
        target = check_tensor(target, [B, L], torch.int, device)
        target_size = check_tensor(target_size, [B], torch.int, device)
        trans_float = check_tensor(trans, [N, N], torch.float, device)

        loss = torch.empty(B, dtype=torch.float, device=device)
        workspace = create_workspace(cls, device, B, T, N, L)
        run_forward(
            cls,
            device,
            B,
            T,
            N,
            L,
            scale_mode,
            get_data_ptr_as_bytes(input_float),
            get_data_ptr_as_bytes(target),
            get_data_ptr_as_bytes(target_size),
            get_data_ptr_as_bytes(trans_float),
            get_data_ptr_as_bytes(loss),
            get_data_ptr_as_bytes(workspace),
        )
        ctx.save_for_backward(input, target, target_size, trans, workspace)
        return loss.to(input)

    @classmethod
    def backward(cls, ctx, grad):
        input, target, target_size, trans, workspace = ctx.saved_tensors
        B = input.size(0)
        T = input.size(1)
        N = input.size(2)
        L = target.size(1)
        device = input.device

        grad = check_tensor(grad, [B], torch.float, device)

        input_grad = torch.empty(B, T, N, dtype=torch.float, device=device)
        trans_grad = torch.empty(N, N, dtype=torch.float, device=device)
        run_backward(
            cls,
            device,
            B,
            T,
            N,
            L,
            get_data_ptr_as_bytes(target),
            get_data_ptr_as_bytes(target_size),
            get_data_ptr_as_bytes(grad),
            get_data_ptr_as_bytes(input_grad),
            get_data_ptr_as_bytes(trans_grad),
            get_data_ptr_as_bytes(workspace),
        )

        return input_grad.to(input), None, None, trans_grad.to(trans), None


class FCCFunction(torch.autograd.Function):
    @staticmethod
    def cuda_impl():
        return _C.CudaFullConnectionCriterion

    @staticmethod
    def cpu_impl():
        return _C.CpuFullConnectionCriterion

    @classmethod
    def forward(cls, ctx, input, target_size, trans, scale_mode):
        B = input.size(0)
        T = input.size(1)
        N = input.size(2)
        device = input.device

        input_float = check_tensor(input, [B, T, N], torch.float, device)
        if scale_mode != _C.CriterionScaleMode.NONE:
            target_size = check_tensor(target_size, [B], torch.int, device)
        trans_float = check_tensor(trans, [N, N], torch.float, device)

        loss = torch.empty(B, dtype=torch.float, device=device)
        workspace = create_workspace(cls, device, B, T, N)
        run_forward(
            cls,
            device,
            B,
            T,
            N,
            scale_mode,
            get_data_ptr_as_bytes(input_float),
            get_data_ptr_as_bytes(target_size),
            get_data_ptr_as_bytes(trans_float),
            get_data_ptr_as_bytes(loss),
            get_data_ptr_as_bytes(workspace),
        )
        ctx.save_for_backward(input, trans, workspace)
        return loss.to(input)

    @classmethod
    def backward(cls, ctx, grad):
        input, trans, workspace = ctx.saved_tensors
        B = input.size(0)
        T = input.size(1)
        N = input.size(2)
        device = input.device

        grad = check_tensor(grad, [B], torch.float, device)

        input_grad = torch.empty(B, T, N, dtype=torch.float, device=device)
        trans_grad = torch.empty(N, N, dtype=torch.float, device=device)
        run_backward(
            cls,
            device,
            B,
            T,
            N,
            get_data_ptr_as_bytes(trans),
            get_data_ptr_as_bytes(grad),
            get_data_ptr_as_bytes(input_grad),
            get_data_ptr_as_bytes(trans_grad),
            get_data_ptr_as_bytes(workspace),
        )
        return input_grad.to(input), None, trans_grad.to(trans), None


class ASGLoss(nn.Module):
    def __init__(self, N, scale_mode=_C.CriterionScaleMode.NONE):
        super().__init__()
        self.trans = nn.Parameter(
            torch.zeros(N, N, dtype=torch.float, requires_grad=True)
        )
        self.scale_mode = scale_mode

    def forward(self, input, target, target_size):
        return FCCFunction.apply(
            input, target_size, self.trans, self.scale_mode
        ) - FACFunction.apply(input, target, target_size, self.trans, self.scale_mode)
