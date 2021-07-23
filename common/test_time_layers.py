from unittest import TestCase
from . import time_layers as TL
import numpy as np
from numpy.testing import assert_equal


class RNNTest(TestCase):
    def test_forward(self):
        N, D, H = 2, 3, 4
        Wx = np.linspace(0, 0.5, 12).reshape(D, H)
        Wh = np.linspace(-0.5, 0.5, H*H).reshape(H, H)
        b = np.zeros(H)
        x = np.array([[1, 2, 1], [2, 3, 2]])
        Rnn = TL.RNN(Wx, Wh, b)
        h_prev = np.ones((N, H))
        h = Rnn.forward(x, h_prev)
        assert_equal(h, np.tanh(h_prev @ Wh + x@Wx + b))

    def test_backward(self):
        N, D, H = 2, 3, 4
        Wx = np.arange(12).reshape(D, H)
        Wh = np.arange(1, 1 + 16).reshape(H, H)
        b = np.arange(H)
        x = np.array([[1, 2, 1], [2, 3, 2]])
        Rnn = TL.RNN(Wx, Wh, b)
        h_prev = np.ones((N, H))

        h = Rnn.forward(x, h_prev)
        dh_next = np.arange(N*H).reshape(N, H)
        dx, dh_prev = Rnn.backward(dh_next)

        dWx, dWh, db = Rnn.grads
        dt = dh_next*(1-h**2)
        assert_equal(db, dt.sum(axis=0))
        assert_equal(dWh, h_prev.T @ dt)
        assert_equal(dWx, x.T @ dt)
        assert_equal(dx, dt @ Wx.T)
        assert_equal(dh_prev, dt @ Wh.T)
