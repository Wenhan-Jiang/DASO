import math
import torch
from torch.optim.optimizer import Optimizer

class DASO(Optimizer):
    r"""Implements Dual averaging with step size optimization for adaptive deep learning.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate α (default: 1e-3)
        momentum (float, optional): momentum parameter (default: 0.9)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_sq_factor (float, optional): factor for scaling gradient square accumulation (default: 2.5e-4)
        amsgrad (bool, optional): whether to use the AMSGrad variant (default: False)
        weight_decouple(bool, optional): whether to use the DASO-W variant (default: False)
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, eps=1e-8, weight_decay=0, grad_sq_factor=2.5e-4,
                 amsgrad=False, weight_decouple=False):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if momentum <= 0:
            raise ValueError(f"Invalid momentum parameter: {momentum}")
        if grad_sq_factor <= 0:
            raise ValueError(f"Invalid gradient square factor: {grad_sq_factor}")

        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay, grad_sq_factor=grad_sq_factor,
                        amsgrad=amsgrad, weight_decouple=weight_decouple)
        super(DASO, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = closure() if closure is not None else None

        if 't' not in self.state:
            self.state['t'] = torch.tensor(0, dtype=torch.long)

        t = self.state['t'].item()
        for group in self.param_groups:
            lr, decay, momentum, eps, grad_sq_factor, amsgrad, weight_decouple = (
                group['lr'], group['weight_decay'], group['momentum'],
                group['eps'], group['grad_sq_factor'], group['amsgrad'], group['weight_decouple']
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state for each parameter
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['grad_sq_avg'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_grad_sq_avg'] = torch.zeros_like(p.data)

                exp_avg, grad_sq_avg = state['exp_avg'], state['grad_sq_avg']
                if amsgrad:
                    max_grad_sq_avg = state['max_grad_sq_avg']

                if weight_decouple:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                else:
                    if decay != 0:
                        grad = grad.add(p.data, alpha=decay)

                # Update moving averages
                exp_avg.mul_(momentum).add_(grad, alpha=(1 - momentum))
                grad_sq_avg.mul_(1 / math.sqrt(t + 1)).addcmul_(grad, grad, value=grad_sq_factor)


                if amsgrad:
                    # Maintain the maximum of all 2nd moment running averages
                    torch.max(max_grad_sq_avg, grad_sq_avg, out=max_grad_sq_avg)
                    denom = max_grad_sq_avg.add(eps).pow(1 / 3)
                else:
                    denom = grad_sq_avg.add(eps).pow(1 / 3)

                # Compute bias-corrected step size
                step_size = lr / (1 - momentum ** (t + 1))

                # Update parameter
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        self.state['t'] += 1
        return loss
