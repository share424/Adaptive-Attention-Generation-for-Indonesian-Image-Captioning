"""This module contains utility functions for PyTorch.

References:
    [1] https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
"""
import torch


def optimizer_to(optim: torch.optim, device: str):
    """Move the optimizer state to the device.
    
    Args:
        optim (torch.optim): Optimizer to move
        device (str): Device to move to
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(sched: torch.optim, device: str):
    """Move the scheduler state to the device.
    
    Args:
        sched (torch.optim): Scheduler to move
        device (str): Device to move to
    """
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


def set_requires_grad(model: torch.nn.Module, requires_grad: bool):
    """Set the requires_grad attribute of a model.
    
    Args:
        model (torch.nn.Module): Model to set
        requires_grad (bool): Value to set
    """
    for param in model.parameters():
        param.requires_grad = requires_grad