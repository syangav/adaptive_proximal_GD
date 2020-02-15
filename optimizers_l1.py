import numpy as np

import numpy.linalg as la

from trainer import Trainer

def prox_l1(v, l1, lr):
    return np.multiply(np.sign(v),np.maximum(0,np.abs(v)-l1*lr))
    # return np.maximum(0,v-l1*lr) - np.maximum(0,-v-l1*lr)


class Gd_l1(Trainer):
    """
    Gradient descent with constant learning rate.

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr, l1, *args, **kwargs):
        super(Gd_l1, self).__init__(*args, **kwargs)
        self.lr = lr
        self.l1 = l1

    def step(self):
        # return self.w - self.lr * self.grad
        return prox_l1(self.w - self.lr * self.grad, self.l1, self.lr)

    def init_run(self, *args, **kwargs):
        super(Gd_l1, self).init_run(*args, **kwargs)


class Ad_grad_l1(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant

    this is algorithm 1 -- adaptive gradient descent --  in the paper if we don't know L
    this is alogrithm 2 -- adaptive GD(L is known) -- then set eps = 1/L^2

    Arguments:
        eps (float): an estimate of 1 / L^2, where L is the global smoothness constant
    """
    def __init__(self, l1, eps=0.0, *args, **kwargs):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        super(Ad_grad_l1, self).__init__(*args, **kwargs)
        self.eps = eps
        self.l1 = l1

    def estimate_stepsize(self):
        L = la.norm(self.grad - self.grad_old) / la.norm(self.w - self.w_old)
        if np.isinf(self.theta):
            lr_new = 0.5 / L
        else:
            lr_new = min(np.sqrt(1 + self.theta) * self.lr, self.eps / self.lr + 0.5 / L)
        self.theta = lr_new / self.lr
        self.lr = lr_new

    def step(self):
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        # return self.w - self.lr * self.grad
        return prox_l1(self.w - self.lr * self.grad, self.l1, self.lr)

    def init_run(self, *args, **kwargs):
        super(Ad_grad_l1, self).init_run(*args, **kwargs)
        self.lrs = []
        self.theta = np.inf
        grad = self.grad_func(self.w)
        # The first estimate is normalized gradient with a small coefficient
        self.lr = 1e-5 / la.norm(grad)
        self.w_old = self.w.copy()
        self.grad_old = grad
        self.w = prox_l1(self.w - self.lr * grad, self.l1, self.lr)
        # self.w -= self.lr * grad
        self.save_checkpoint()

    def update_logs(self):
        super(Ad_grad_l1, self).update_logs()
        self.lrs.append(self.lr)
