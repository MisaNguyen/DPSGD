import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim import Optimizer

"""
UPDATE GRADIENT NORM IN THE OPTIMIZER
"""
EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

class _GNScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_gn', group['gn'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_gn' not in group:
                    raise KeyError("param 'initial_gn' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_gns = [group['initial_gn'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_gn(self):
        """ Return last computed gradient norm by current scheduler.
        """
        return self._last_gn

    def get_gn(self):
        # Compute Gradient Norm using chainable form of the scheduler
        raise NotImplementedError

    def print_gn(self, is_verbose, group, gradient_norm, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, gradient_norm))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, group, gradient_norm))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after gradient norm scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `gn_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_gn_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_gn_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_gn_called_within_step = False

        with _enable_get_gn_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_gn()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_gn"):
                    values = self._get_closed_form_gn()
                else:
                    values = self.get_gn()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, gn = data
            param_group['gn'] = gn
            self.print_gn(self.verbose, i, gn, epoch)

        self._last_gn = [group['gn'] for group in self.optimizer.param_groups]


class LambdaGN(_GNScheduler):
    """Sets the learning rate of each parameter group to the initial gn
    times a given function. When last_epoch=-1, sets initial gn as gn.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaGN(optimizer, gn_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, gn_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(gn_lambda, list) and not isinstance(gn_lambda, tuple):
            self.lr_lambdas = [gn_lambda] * len(optimizer.param_groups)
        else:
            if len(gn_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(gn_lambda)))
            self.lr_lambdas = list(gn_lambda)
        super(LambdaGN, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'gn_lambdas')}
        state_dict['gn_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['gn_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('gn_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['gn_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.gn_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_gn_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_gn * lmbda(self.last_epoch)
                for lmbda, base_gn in zip(self.lr_lambdas, self.base_gns)]


class StepGN(_GNScheduler):
    """Decays the gradient norm of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial gn as gn.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gradient_norm (float): Period of gradient decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses gn = 0.05 for all groups
        >>> # gn = 0.05     if epoch < 30
        >>> # gn = 0.005    if 30 <= epoch < 60
        >>> # gn = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepGN(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, gradient_norm, gamma=0.1, last_epoch=-1, verbose=False):
        self.gradient_norm = gradient_norm
        self.gamma = gamma
        super(StepGN, self).__init__(optimizer, last_epoch, verbose)

    def get_gn(self):
        if not self._get_gn_called_within_step:
            warnings.warn("To get the last gradient norm computed by the scheduler, "
                          "please use `get_last_gn()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.gradient_norm != 0):
            return [group['gn'] for group in self.optimizer.param_groups]
        return [group['gn'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_gn(self):
        return [base_gn * self.gamma ** (self.last_epoch // self.gradient_norm)
                for base_gn in self.base_gns]

"""
UPDATE GRADIENT NORM NORMALLY (OUTSIDE OF THE OPTIMIZER)
"""

class StepGN_normal():
    def __init__(self, gradient_norm, gamma=0.1, epoch = -1):
        self.gradient_norm = gradient_norm
        self.gamma = gamma
        self.epoch = epoch

    def get_gn_after_epochs(self):
        if(self.epoch < 0):
            warnings.warn("epoch can not be set to negative", UserWarning)
        # Gradient_norm = base gradient_norm * gamma^epoch
        return self.gradient_norm * (self.gamma ** self.epoch)
