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

class _NMScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_nm', group['nm'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_nm' not in group:
                    raise KeyError("param 'initial_nm' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_nms = [group['initial_nm'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `nm_scheduler.step()` is called after
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

    def get_last_nm(self):
        """ Return last computed gradient norm by current scheduler.
        """
        return self._last_nm

    def get_nm(self):
        # Compute Gradient Norm using chainable form of the scheduler
        raise NotImplementedError

    def print_nm(self, is_verbose, group, noise_multiplier, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, noise_multiplier))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, group, noise_multiplier))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after gradient norm scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "nm_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first nm_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `nm_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `nm_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_nm_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_nm_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_nm_called_within_step = False

        with _enable_get_nm_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_nm()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_nm"):
                    values = self._get_closed_form_nm()
                else:
                    values = self.get_nm()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, nm = data
            param_group['nm'] = nm
            self.print_nm(self.verbose, i, nm, epoch)

        self._last_nm = [group['nm'] for group in self.optimizer.param_groups]


class LambdaNM(_NMScheduler):
    """Sets the learning rate of each parameter group to the initial nm
    times a given function. When last_epoch=-1, sets initial nm as nm.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        nm_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaNM(optimizer, nm_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, nm_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(nm_lambda, list) and not isinstance(nm_lambda, tuple):
            self.nm_lambdas = [nm_lambda] * len(optimizer.param_groups)
        else:
            if len(nm_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} nm_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(nm_lambda)))
            self.nm_lambdas = list(nm_lambda)
        super(LambdaNM, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'nm_lambdas')}
        state_dict['nm_lambdas'] = [None] * len(self.nm_lambdas)

        for idx, fn in enumerate(self.nm_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['nm_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        nm_lambdas = state_dict.pop('nm_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['nm_lambdas'] = nm_lambdas

        for idx, fn in enumerate(nm_lambdas):
            if fn is not None:
                self.nm_lambdas[idx].__dict__.update(fn)

    def get_nm(self):
        if not self._get_nm_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_nm()`.")

        return [base_nm * lmbda(self.last_epoch)
                for lmbda, base_nm in zip(self.nm_lambdas, self.base_nms)]


class StepNM(_NMScheduler):
    """Decays the gradient norm of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial nm as nm.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        noise_multiplier (float): Period of gradient decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming optimizer uses nm = 0.05 for all groups
        >>> # nm = 0.05     if epoch < 30
        >>> # nm = 0.005    if 30 <= epoch < 60
nm        >>> # ...
        >>> scheduler = StepNM(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, noise_multiplier, gamma=0.1, last_epoch=-1, verbose=False):
        self.noise_multiplier = noise_multiplier
        self.gamma = gamma
        super(StepNM, self).__init__(optimizer, last_epoch, verbose)

    def get_nm(self):
        if not self._get_nm_called_within_step:
            warnings.warn("To get the last noise multiplier computed by the scheduler, "
                          "please use `get_last_nm()`.", UserWarning)

        if (self.last_epoch == 0) or (self.last_epoch % self.noise_multiplier != 0):
            return [group['nm'] for group in self.optimizer.param_groups]
        return [group['nm'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_nm(self):
        return [base_nm * self.gamma ** (self.last_epoch // self.noise_multiplier)
                for base_nm in self.base_nms]

"""
UPDATE GRADIENT NORM NORMALLY (OUTSIDE OF THE OPTIMIZER)
"""

class StepNM_normal():
    def __init__(self, noise_multiplier, gamma=0.1, epoch = -1):
        self.noise_multiplier = noise_multiplier
        self.gamma = gamma
        self.epoch = epoch

    def get_nm_after_epochs(self):
        if(self.epoch < 0):
            warnings.warn("epoch can not be set to negative", UserWarning)
        # noise_multiplier = base noise_multiplier * gamma^epoch
        return self.noise_multiplier * (self.gamma ** self.epoch)
