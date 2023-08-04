# Third party code
#
# The following code are copied or modified from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

import numpy as np


class Optimizer(object):
    def __init__(self, parameter_size):
        self.dim = parameter_size
        self.t = 1

    def update(self, theta, global_g):
        self.t += 1
        step = self._compute_step(global_g)
        tmp = np.linalg.norm(theta)
        if tmp == 0.0:
            tmp = 1.0
        ratio = np.linalg.norm(step) / tmp
        return theta + step, ratio

    def _compute_step(self, global_g):
        raise NotImplementedError

    def get_stepsize(self):
        return self.stepsize

    def update_stepsize(self, stepsize):
        self.stepsize = stepsize

    def update_new(self, theta, global_g):
        self.t += 1
        step = self._compute_step(global_g)
        return theta + step


class SGD(Optimizer):
    def __init__(self, parameter_size, stepsize):
        Optimizer.__init__(self, parameter_size)
        self.stepsize = stepsize

    def _compute_step(self, global_g):
        step = -self.stepsize * global_g
        return step


class Momentum(Optimizer):
    """Momentum optimizer"""
    def __init__(self, parameter_size, stepsize, momentum=0.9):
        Optimizer.__init__(self, parameter_size)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, global_g):
        self.v = self.momentum * self.v + (1. - self.momentum) * global_g
        step = -self.stepsize * self.v
        return step


class CorrectMomentum(Optimizer):
    """
    Momentum optimizer with bias-corrected estimates on v
    A correct denominator is added to solve the problem that
        the step is small in the initial state due to the momentum.
    I.e. v = beta*v + (1 - beta)*g
         step = -alpha * v / (1 - beta**t)
    """
    def __init__(self, parameter_size, stepsize, momentum=0.9):
        Optimizer.__init__(self, parameter_size)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, global_g):
        self.v = self.momentum * self.v + (1. - self.momentum) * global_g
        step = - self.stepsize * self.v / (1 - self.momentum ** self.t)
        return step


class VectorMomentum(Optimizer):
    """
    Momentum optimizer with vector momentum
    I.e. if t == 1:
            v = g
        else:
            v = beta*v + sqrt(1 - beta**2)*g
            step = -alpha * v
    """
    def __init__(self, parameter_size, stepsize, momentum=0.5):
        Optimizer.__init__(self, parameter_size)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, global_g):
        if self.t == 1:
            self.v = global_g
        else:
            self.v = self.momentum * self.v + np.sqrt(1. - self.momentum ** 2) * global_g
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self,
                 parameter_size,
                 stepsize,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08):
        Optimizer.__init__(self, parameter_size)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, global_g):
        a = self.stepsize * (np.sqrt(1 - self.beta2**self.t) /
                             (1 - self.beta1**self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * global_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (global_g * global_g)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
