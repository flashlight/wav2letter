#!/usr/bin/env python3

import struct
import sys

import torch
import wav2letter._criterion as _C


def get_data_ptr_as_bytes(tensor):
    return struct.pack("P", tensor.data_ptr())


def get_cuda_stream_as_bytes():
    s = torch.cuda.current_stream().cuda_stream
    return s.to_bytes(_C.sizeof_cuda_stream, byteorder=sys.byteorder)


def check_tensor(name, tensor, size, dtype, device):
    shape = torch.Size(size)
    if tensor.shape != shape:
        raise ValueError(
            f"wrong size for {name} tensor: expected {shape}, got {tensor.shape}"
        )
    elif tensor.dtype != dtype:
        raise ValueError(
            f"wrong type for {name} tensor: expected {dtype}, got {tensor.dtype}"
        )
    elif tensor.device != device:
        raise ValueError(
            f"wrong device for {name} tensor: expected {device}, got {tensor.device}"
        )
    elif not tensor.is_contiguous():
        raise ValueError(f"{name} tensor is not contiguous")


class FACFunction(torch.autograd.Function):
    @staticmethod
    def select_criterion(is_cuda):
        return (
            _C.CudaForceAlignmentCriterion if is_cuda else _C.CpuForceAlignmentCriterion
        )

    @classmethod
    def forward(cls, ctx, input, target, target_size, trans, scale_mode):
        device = input.device
        criterion = cls.select_criterion(input.is_cuda)

        B = input.size(0)
        T = input.size(1)
        N = input.size(2)
        L = target.size(1)

        check_tensor("input", input, [B, T, N], torch.float, device)
        check_tensor("target", target, [B, L], torch.int, device)
        check_tensor("target_size", target_size, [B], torch.int, device)
        check_tensor("trans", trans, [N, N], torch.float, device)

        loss = torch.empty(B, dtype=torch.float, device=device)
        workspace = torch.empty(
            criterion.getWorkspaceSize(B, T, N, L), dtype=torch.uint8, device=device
        )
        args = [
            B,
            T,
            N,
            L,
            scale_mode,
            get_data_ptr_as_bytes(input),
            get_data_ptr_as_bytes(target),
            get_data_ptr_as_bytes(target_size),
            get_data_ptr_as_bytes(trans),
            get_data_ptr_as_bytes(loss),
            get_data_ptr_as_bytes(workspace),
        ]
        if input.is_cuda:
            args.append(get_cuda_stream_as_bytes())

        criterion.forward(*args)
        ctx.save_for_backward(input, target, target_size, trans, workspace)
        return loss

    @classmethod
    def backward(cls, ctx, grad):
        input, target, target_size, trans, workspace = ctx.saved_tensors
        device = input.device
        criterion = cls.select_criterion(input.is_cuda)

        B = input.size(0)
        T = input.size(1)
        N = input.size(2)
        L = target.size(1)

        check_tensor("grad", grad, [B], torch.float, device)

        inputGrad = torch.empty(B, T, N, dtype=torch.float, device=device)
        transGrad = torch.empty(N, N, dtype=torch.float, device=device)
        args = [
            B,
            T,
            N,
            L,
            get_data_ptr_as_bytes(target),
            get_data_ptr_as_bytes(target_size),
            get_data_ptr_as_bytes(grad),
            get_data_ptr_as_bytes(inputGrad),
            get_data_ptr_as_bytes(transGrad),
            get_data_ptr_as_bytes(workspace),
        ]
        if input.is_cuda:
            args.append(get_cuda_stream_as_bytes())

        criterion.backward(*args)
        return inputGrad, None, None, transGrad, None


class FCCFunction(torch.autograd.Function):
    @staticmethod
    def select_criterion(is_cuda):
        return (
            _C.CudaFullConnectionCriterion if is_cuda else _C.CpuFullConnectionCriterion
        )

    @classmethod
    def forward(cls, ctx, input, target_size, trans, scale_mode):
        device = input.device
        criterion = cls.select_criterion(input.is_cuda)

        B = input.size(0)
        T = input.size(1)
        N = input.size(2)

        check_tensor("input", input, [B, T, N], torch.float, device)
        if scale_mode != _C.CriterionScaleMode.NONE:
            check_tensor("target_size", target_size, [B], torch.int, device)
        check_tensor("trans", trans, [N, N], torch.float, device)

        loss = torch.empty(B, dtype=torch.float, device=device)
        workspace = torch.empty(
            criterion.getWorkspaceSize(B, T, N), dtype=torch.uint8, device=device
        )
        args = [
            B,
            T,
            N,
            scale_mode,
            get_data_ptr_as_bytes(input),
            get_data_ptr_as_bytes(target_size),
            get_data_ptr_as_bytes(trans),
            get_data_ptr_as_bytes(loss),
            get_data_ptr_as_bytes(workspace),
        ]
        if input.is_cuda:
            args.append(get_cuda_stream_as_bytes())

        criterion.forward(*args)
        ctx.save_for_backward(input, trans, workspace)
        return loss

    @classmethod
    def backward(cls, ctx, grad):
        input, trans, workspace = ctx.saved_tensors
        device = input.device
        criterion = cls.select_criterion(input.is_cuda)

        B = input.size(0)
        T = input.size(1)
        N = input.size(2)

        check_tensor("grad", grad, [B], torch.float, device)

        inputGrad = torch.empty(B, T, N, dtype=torch.float, device=device)
        transGrad = torch.empty(N, N, dtype=torch.float, device=device)
        args = [
            B,
            T,
            N,
            get_data_ptr_as_bytes(trans),
            get_data_ptr_as_bytes(grad),
            get_data_ptr_as_bytes(inputGrad),
            get_data_ptr_as_bytes(transGrad),
            get_data_ptr_as_bytes(workspace),
        ]
        if input.is_cuda:
            args.append(get_cuda_stream_as_bytes())

        criterion.backward(*args)
        return inputGrad, None, transGrad, None


class ASGLoss(torch.nn.Module):
    def __init__(self, N, scale_mode=_C.CriterionScaleMode.NONE):
        super().__init__()
        self.trans = torch.nn.Parameter(
            torch.zeros(N, N, dtype=torch.float, requires_grad=True)
        )
        self.scale_mode = scale_mode

    def forward(self, input, target, target_size):
        return FCCFunction.apply(
            input, target_size, self.trans, self.scale_mode
        ) - FACFunction.apply(input, target, target_size, self.trans, self.scale_mode)
