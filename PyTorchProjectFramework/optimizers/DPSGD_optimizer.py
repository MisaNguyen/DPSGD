# from __future__ import annotations

import torch.optim as optim
import torch
from torch.optim import Optimizer
from torch import Tensor
from typing import List, Optional

import numpy as np



# import logging
# from typing import Callable, List, Optional, Union
#
# import torch
# from opacus.optimizers.utils import params
# from torch import nn
# from torch.optim import Optimizer

def generate_noise(dimension=1, C=1.0, sigma=1.0):
    # 'the clip bound of the gradients'
    # clip_bound = (0.5)

    # 'sigma'
    # sigma = (1.0)

    # adjacency matrix with one more tuple
    # sensitivity = clip_bound

    # w_noise = np.full((dimension,1), 0)
    # for randIndex in range(dimension):
    #     noise = np.random.normal(0.0, sigma * (sensitivity**2), 1)
    #     w_noise[randIndex,:] = noise

    # Ref: https://www.sharpsightlabs.com/blog/numpy-random-normal/
    noise = np.random.normal(0.0, sigma * (C**2), size=(dimension,1))
    # w_noise = noise

    return noise

# def new_optimizer
# https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/torch/optim/_functional.py
def dpsgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        noise_multiplier: float,
          max_grad_norm: float,):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """
    # input("Here")
    # input(d_p_list)
    # print("HERE")
    # input(len(d_p_list))
    # input(d_p_list[0].shape)
    # print("END")

    for i, param in enumerate(params):
        # input(param.shape)
        # input(len(params))
        d_p = d_p_list[i]
        # If we use weight decay
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        # If we use momentum
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf
            #Add dpsgd condition:
        # print("Before",d_p)
        # input(torch.normal(mean=torch.Tensor([0.0]),
        #                    std=noise_multiplier * max_grad_norm).to(device=torch.device("cuda:0")))
        # input(d_p)
        d_p += torch.normal(mean=torch.Tensor([0.0]),
                                   std=noise_multiplier * max_grad_norm).to(device=torch.device("cuda:0"))
        # input(d_p.shape)
        # print("after",d_p)
        # input("HERE")
        # input(param)
        param.add_(d_p, alpha=-lr)
        # param.grad = None  # Reset for next iteratio
        # # Clipping
        # per_sample_grad = d_p.detach().clone()
        # torch.nn.utils.clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
        # param.accumulated_grads.append(per_sample_grad)

    # for param in params:
    #     param.grad = torch.stack(param.accumulated_grads, dim=0)

    # for param in params:
    #     # Add the noise
    #     param.grad += torch.normal(mean=torch.Tensor([0.0]),
    #                                std=noise_multiplier * max_grad_norm).to(device=torch.device("cuda:0"))
    #     # input(param.shape)
    #     # input(param.grad.shape)
    #     # input(param.grad)
    #     # Gradient descent step
    #     # Gradient descent step
    #     param.add_(param.grad, alpha=-lr)
    #     # param += torch.normal(mean=torch.Tensor([0.0]), std=args.noise_multiplier * args.max_grad_norm).to(device=torch.device("cuda:0"))
    #
    #     param.grad = None  # Reset for next iteration



    # for i, param in enumerate(params):

        # noise = ((C*sigma)**2)*torch.randn(d_p.shape)
        # input(noise.shape)
        # input(noise)
        # input(d_p.shape)
        # input(param.shape)
        # Note: add_ (in_place) = similar to +=
        # get gradient's shape
        # grad_shape = d_p_list[i].shape
        # Gaussian Distribution: dist = N(0,C^2 sigma^2)
        # dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([(C*sigma)**2]))
        # dist = torch.distributions.normal.Normal(0.0, (max_grad_norm*noise_multiplier))

        # Create noise tensor
        # noise = dist.rsample(grad_shape).to(device=torch.device("cuda:0"))
        # noise = torch.normal(mean=0.0,
        #                      std=max_grad_norm*noise_multiplier).to(device=torch.device("cuda:0"))

        # input(d_p_list[0].shape)
        # input(d_p_list[0])
        # input(d_p_list[1].shape)
        # input(d_p_list[1])
        # input(d_p_list[2].shape)
        # input(d_p_list[2])
        # param.grad = torch.cat(d_p_list, dim=0)

        # input(d_p.shape)
        # input(noise.shape)
        # d_p = d_p + noise
        # input((C*sigma)**2)
        # print("Grad:")
        # input(d_p)
        # print("noise")
        # input(noise)
        # input()
        # input(noise/len(d_p))
        # input(d_p)
        # d_p.add_(noise/len(d_p))
        # input(param.shape)
        # input(d_p.shape)
        # input(param)
        # input(noise)

        # d_p.add_(noise)
        # input(param)
        # param = param - d_p*lr
    # for i, param in enumerate(params):
    #     param.grad += torch.normal(mean=torch.Tensor([0.0]),
    #                                std=noise_multiplier * max_grad_norm).to(device=torch.device("cuda:0"))
    #     param.add_(param.grad, alpha=-lr)
        # param.grad



# http://pytorch.org/docs/master/_modules/torch/optim/sgd.html#SGD

class DPSGD(Optimizer):

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, noise_multiplier=1.0, max_grad_norm=1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if noise_multiplier < 0.0:
            raise ValueError("Invalid sigma value: {}".format(noise_multiplier))
        if max_grad_norm < 0.0:
            raise ValueError("Invalid gradient_norm value: {}".format(max_grad_norm))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize,
                        noise_multiplier=noise_multiplier,max_grad_norm=max_grad_norm)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DPSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DPSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # input(group.keys())
            # input(group.keys())
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']
            noise_multiplier = group['noise_multiplier']
            max_grad_norm = group['max_grad_norm']
            # input(group.keys())
            for p in group['params']:
                # input(len(p))
                if p.grad is not None:
                    params_with_grad.append(p)
                    # Clipping norm
                    # d_p = d_p/max(1, torch.linalg.norm(d_p)/C)
                    # per_sample_grad = p.grad.detach().clone()
                    # torch.nn.utils.clip_grad_norm_(per_sample_grad, max_grad_norm, norm_type=2.0, error_if_nonfinite=False)

                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            dpsgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  noise_multiplier=noise_multiplier,
                  max_grad_norm=max_grad_norm,)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


"""
Optimizer callers
"""
def Adadelta_optimizer(model_parameters,learning_rate):
    optimizer = optim.Adadelta(model_parameters, lr=learning_rate)
    return optimizer

# Note: Should choose small learning_rate
def SGD_optimizer(model_parameters,learning_rate):
    optimizer = optim.SGD(model_parameters, lr=learning_rate)
    return optimizer


# def custom_SGD_optimizer(learning_rate,model):
#     pass
# # DPSGD optimizer
def DPSGD_optimizer(model_parameters,learning_rate,noise_multiplier,max_grad_norm):
    optimizer = DPSGD(model_parameters, lr=learning_rate,noise_multiplier=noise_multiplier,max_grad_norm=max_grad_norm)
    return optimizer


"""
Test
"""


# logger = logging.getLogger(__name__)
#
#
# def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
#     """
#     Marks parameters that have already been used in the optimizer step.
#     DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
#     no gradient can be used twice - client must call .zero_grad() between
#     optimizer steps, otherwise privacy guarantees are compromised.
#     This method marks tensors that have already been used in optimizer steps to then
#     check if zero_grad has been duly called.
#     Notes:
#           This is used to only mark ``p.grad_sample`` and ``p.summed_grad``
#     Args:
#         obj: tensor or a list of tensors to be marked
#     """
#
#     if isinstance(obj, torch.Tensor):
#         obj._processed = True
#     elif isinstance(obj, list):
#         for x in obj:
#             x._processed = True
#
#
# def _check_processed_flag_tensor(x: torch.Tensor):
#     """
#     Checks if this gradient tensor has been previously used in optimization step.
#     See Also:
#         :meth:`~opacus.optimizers.optimizer._mark_as_processed`
#     Args:
#         x: gradient tensor
#     Raises:
#         ValueError
#             If tensor has attribute ``._processed`` previously set by
#             ``_mark_as_processed`` method
#     """
#
#     if hasattr(x, "_processed"):
#         raise ValueError(
#             "Gradients haven't been cleared since the last optimizer step. "
#             "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
#             "on each step"
#         )
#
#
# def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
#     """
#     Checks if this gradient tensor (or a list of tensors) has been previously
#     used in optimization step.
#     See Also:
#         :meth:`~opacus.optimizers.optimizer._mark_as_processed`
#     Args:
#         x: gradient tensor or a list of tensors
#     Raises:
#         ValueError
#             If tensor (or at least one tensor from the list) has attribute
#             ``._processed`` previously set by ``_mark_as_processed`` method
#     """
#
#     if isinstance(obj, torch.Tensor):
#         _check_processed_flag_tensor(obj)
#     elif isinstance(obj, list):
#         for x in obj:
#             _check_processed_flag_tensor(x)
#
#
# def _generate_noise(
#         std: float,
#         reference: torch.Tensor,
#         generator=None,
#         secure_mode: bool = False,
# ) -> torch.Tensor:
#     """
#     Generates noise according to a Gaussian distribution with mean 0
#     Args:
#         std: Standard deviation of the noise
#         reference: The reference Tensor to get the appropriate shape and device
#             for generating the noise
#         generator: The PyTorch noise generator
#         secure_mode: boolean showing if "secure" noise need to be generate
#             (see the notes)
#     Notes:
#         If `secure_mode` is enabled, the generated noise is also secure
#         against the floating point representation attacks, such as the ones
#         in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
#         The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
#         The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
#         achieved through calling the Gaussian noise function 2*n times, when n=2
#         (see section 5.1 in https://arxiv.org/abs/2107.10138).
#         Reason for choosing n=2: n can be any number > 1. The bigger, the more
#         computation needs to be done (`2n` Gaussian samples will be generated).
#         The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
#         is not really necessary. The complexity of the attack is `2^p(2n-1)`.
#         In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
#         `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
#         enough for an attacker to break.
#     """
#     zeros = torch.zeros(reference.shape, device=reference.device)
#     if std == 0:
#         return zeros
#     # TODO: handle device transfers: generator and reference tensor
#     # could be on different devices
#     if secure_mode:
#         torch.normal(
#             mean=0,
#             std=std,
#             size=(1, 1),
#             device=reference.device,
#             generator=generator,
#         )  # generate, but throw away first generated Gaussian sample
#         sum = zeros
#         for _ in range(4):
#             sum += torch.normal(
#                 mean=0,
#                 std=std,
#                 size=reference.shape,
#                 device=reference.device,
#                 generator=generator,
#             )
#         return sum / 2
#     else:
#         return torch.normal(
#             mean=0,
#             std=std,
#             size=reference.shape,
#             device=reference.device,
#             generator=generator,
#         )
#
#
# def _get_flat_grad_sample(p: torch.Tensor):
#     """
#     Return parameter's per sample gradients as a single tensor.
#     By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
#     batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
#     only one batch, and a list of tensors if gradients are accumulated over multiple
#     steps. This is done to provide visibility into which sample belongs to which batch,
#     and how many batches have been processed.
#     This method returns per sample gradients as a single concatenated tensor, regardless
#     of how many batches have been accumulated
#     Args:
#         p: Parameter tensor. Must have ``grad_sample`` attribute
#     Returns:
#         ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
#         concatenating every tensor in ``p.grad_sample`` if it's a list
#     Raises:
#         ValueError
#             If ``p`` is missing ``grad_sample`` attribute
#     """
#
#     if not hasattr(p, "grad_sample"):
#         raise ValueError(
#             "Per sample gradient not found. Are you using GradSampleModule?"
#         )
#     if isinstance(p.grad_sample, torch.Tensor):
#         return p.grad_sample
#     elif isinstance(p.grad_sample, list):
#         return torch.cat(p.grad_sample, dim=0)
#     else:
#         raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")
# class DPOptimizer(Optimizer):
#     """
#     ``torch.optim.Optimizer`` wrapper that adds additional functionality to clip per
#     sample gradients and add Gaussian noise.
#     Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
#     ``DPOptimzer`` assumes that parameters over which it performs optimization belong
#     to GradSampleModule and therefore have the ``grad_sample`` attribute.
#     On a high level ``DPOptimizer``'s step looks like this:
#     1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms
#     2) Clip ``p.grad_sample`` so that per sample norm is not above threshold
#     3) Aggregate clipped per sample gradients into ``p.grad``
#     4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
#     max grad norm limit (``std = noise_multiplier * max_grad_norm``).
#     5) Call underlying optimizer to perform optimization step
#     Examples:
#         >>> module = MyCustomModel()
#         >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
#         >>> dp_optimzer = DPOptimizer(
#         ...     optimizer=optimizer,
#         ...     noise_multiplier=1.0,
#         ...     max_grad_norm=1.0,
#         ...     expected_batch_size=4,
#         ... )
#     """
#
#     def __init__(
#             self,
#             optimizer: Optimizer,
#             *,
#             noise_multiplier: float,
#             max_grad_norm: float,
#             expected_batch_size: Optional[int],
#             loss_reduction: str = "mean",
#             generator=None,
#             secure_mode: bool = False,
#     ):
#         """
#         Args:
#             optimizer: wrapped optimizer.
#             noise_multiplier: noise multiplier
#             max_grad_norm: max grad norm used for gradient clipping
#             expected_batch_size: batch_size used for averaging gradients. When using
#                 Poisson sampling averaging denominator can't be inferred from the
#                 actual batch size. Required is ``loss_reduction="mean"``, ignored if
#                 ``loss_reduction="sum"``
#             loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
#                 is a sum or a mean operation. Can take values "sum" or "mean"
#             generator: torch.Generator() object used as a source of randomness for
#                 the noise
#             secure_mode: if ``True`` uses noise generation approach robust to floating
#                 point arithmetic attacks.
#                 See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
#         """
#         if loss_reduction not in ("mean", "sum"):
#             raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")
#
#         if loss_reduction == "mean" and expected_batch_size is None:
#             raise ValueError(
#                 "You must provide expected batch size of the loss reduction is mean"
#             )
#
#         self.original_optimizer = optimizer
#         self.noise_multiplier = noise_multiplier
#         self.max_grad_norm = max_grad_norm
#         self.loss_reduction = loss_reduction
#         self.expected_batch_size = expected_batch_size
#         self.step_hook = None
#         self.generator = generator
#         self.secure_mode = secure_mode
#
#         self.param_groups = optimizer.param_groups
#         self.state = optimizer.state
#         self._step_skip_queue = []
#         self._is_last_step_skipped = False
#
#         for p in self.params:
#             p.summed_grad = None
#
#     def signal_skip_step(self, do_skip=True):
#         """
#         Signals the optimizer to skip an optimization step and only perform clipping and
#         per sample gradient accumulation.
#         On every call of ``.step()`` optimizer will check the queue of skipped step
#         signals. If non-empty and the latest flag is ``True``, optimizer will call
#         ``self.clip_and_accumulate``, but won't proceed to adding noise and performing
#         the actual optimization step.
#         It also affects the behaviour of ``zero_grad()``. If the last step was skipped,
#         optimizer will clear per sample gradients accumulated by
#         ``self.clip_and_accumulate`` (``p.grad_sample``), but won't touch aggregated
#         clipped gradients (``p.summed_grad``)
#         Used by :class:`~opacus.utils.batch_memory_manager.BatchMemoryManager` to
#         simulate large virtual batches with limited memory footprint.
#         Args:
#             do_skip: flag if next step should be skipped
#         """
#         self._step_skip_queue.append(do_skip)
#
#     def _check_skip_next_step(self):
#         if self._step_skip_queue:
#             return self._step_skip_queue.pop(0)
#         else:
#             return False
#
#     @property
#     def params(self) -> List[nn.Parameter]:
#         """
#         Returns a flat list of ``nn.Parameter`` managed by the optimizer
#         """
#         return params(self)
#
#     @property
#     def grad_samples(self) -> List[torch.Tensor]:
#         """
#         Returns a flat list of per sample gradient tensors (one per parameter)
#         """
#         ret = []
#         for p in self.params:
#             ret.append(_get_flat_grad_sample(p))
#         return ret
#
#     @property
#     def accumulated_iterations(self) -> int:
#         """
#         Returns number of batches currently accumulated and not yet processed.
#         In other words ``accumulated_iterations`` tracks the number of forward/backward
#         passed done in between two optimizer steps. The value would typically be 1,
#         but there are possible exceptions.
#         Used by privacy accountants to calculate real sampling rate.
#         """
#         vals = []
#         for p in self.params:
#             if not hasattr(p, "grad_sample"):
#                 raise ValueError(
#                     "Per sample gradient not found. Are you using GradSampleModule?"
#                 )
#             if isinstance(p.grad_sample, torch.Tensor):
#                 vals.append(1)
#             elif isinstance(p.grad_sample, list):
#                 vals.append(len(p.grad_sample))
#             else:
#                 raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")
#
#         if len(set(vals)) > 1:
#             raise ValueError(
#                 "Number of accumulated steps is inconsistent across parameters"
#             )
#         return vals[0]
#
#     def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
#         """
#         Attaches a hook to be executed after gradient clipping/noising, but before the
#         actual optimization step.
#         Most commonly used for privacy accounting.
#         Args:
#             fn: hook function. Expected signature: ``foo(optim: DPOptimizer)``
#         """
#
#         self.step_hook = fn
#
#     def clip_and_accumulate(self):
#         """
#         Performs gradient clipping.
#         Stores clipped and aggregated gradients into `p.summed_grad```
#         """
#
#         per_param_norms = [
#             g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
#         ]
#         per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
#         per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
#             max=1.0
#         )
#
#         for p in self.params:
#             _check_processed_flag(p.grad_sample)
#
#             grad_sample = _get_flat_grad_sample(p)
#             grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
#
#             if p.summed_grad is not None:
#                 p.summed_grad += grad
#             else:
#                 p.summed_grad = grad
#
#             _mark_as_processed(p.grad_sample)
#
#     def add_noise(self):
#         """
#         Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
#         """
#
#         for p in self.params:
#             _check_processed_flag(p.summed_grad)
#
#             noise = _generate_noise(
#                 std=self.noise_multiplier * self.max_grad_norm,
#                 reference=p.summed_grad,
#                 generator=self.generator,
#                 secure_mode=self.secure_mode,
#             )
#             p.grad = p.summed_grad + noise
#
#             _mark_as_processed(p.summed_grad)
#
#     def scale_grad(self):
#         """
#         Applies given ``loss_reduction`` to ``p.grad``.
#         Does nothing if ``loss_reduction="sum"``. Divides gradients by
#         ``self.expected_batch_size`` if ``loss_reduction="mean"``
#         """
#         if self.loss_reduction == "mean":
#             for p in self.params:
#                 p.grad /= self.expected_batch_size * self.accumulated_iterations
#
#     def zero_grad(self, set_to_none: bool = False):
#         """
#         Clear gradients.
#         Clears ``p.grad``, ``p.grad_sample`` and ``p.summed_grad`` for all of it's parameters
#         Notes:
#             ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
#             ``p.summed_grad`` is never zeroed out and always set to None.
#             Normal grads can do this, because their shape is always the same.
#             Grad samples do not behave like this, as we accumulate gradients from different
#             batches in a list
#         Args:
#             set_to_none: instead of setting to zero, set the grads to None. (only
#             affects regular gradients. Per sample gradients are always set to None)
#         """
#
#         if set_to_none is False:
#             logger.info(
#                 "Despite set_to_none is set to False, "
#                 "opacus will set p.grad_sample and p.summed_grad to None due to "
#                 "non-trivial gradient accumulation behaviour"
#             )
#
#         for p in self.params:
#             p.grad_sample = None
#
#             if not self._is_last_step_skipped:
#                 p.summed_grad = None
#
#         self.original_optimizer.zero_grad(set_to_none)
#
#     def pre_step(
#             self, closure: Optional[Callable[[], float]] = None
#     ) -> Optional[float]:
#         """
#         Perform actions specific to ``DPOptimizer`` before calling
#         underlying  ``optimizer.step()``
#         Args:
#             closure: A closure that reevaluates the model and
#                 returns the loss. Optional for most optimizers.
#         """
#         self.clip_and_accumulate()
#         if self._check_skip_next_step():
#             self._is_last_step_skipped = True
#             return False
#
#         self.add_noise()
#         self.scale_grad()
#
#         if self.step_hook:
#             self.step_hook(self)
#
#         self._is_last_step_skipped = False
#         return True
#
#     def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
#         if closure is not None:
#             with torch.enable_grad():
#                 closure()
#
#         if self.pre_step():
#             return self.original_optimizer.step(closure)
#         else:
#             return None
#
#     def __repr__(self):
#         return self.original_optimizer.__repr__()
#
#     def state_dict(self):
#         return self.original_optimizer.state_dict()
#
#     def load_state_dict(self, state_dict) -> None:
#         self.original_optimizer.load_state_dict(state_dict)

