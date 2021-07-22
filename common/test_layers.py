from unittest import TestCase
from . import layers as L
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal


class EmbeddingTest(TestCase):
    def test_forward(self):
        W = np.arange(12).reshape(3, 4)
        embedding = L.Embedding(W)
        out = embedding.forward([2, 0])
        assert_equal(out, W[[2, 0]])

    def test_backward(self):
        W = np.arange(12).reshape(3, 4)
        embedding = L.Embedding(W)
        embedding.forward([2, 0])
        embedding.backward(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        dout = np.array([[2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]])
        assert_equal(embedding.grads[0], dout)


class EmbeddingDotTest(TestCase):
    def test_forward(self):
        W = np.arange(12).reshape(3, 4).T
        embDot = L.EmbeddingDot(W)
        h = np.array([[0, 1, 2], [3, 4, 5]])
        idx = np.array([3, 1])
        out = embDot.forward(h, idx)
        assert_equal(out, [[29, 23], [92, 68]])
