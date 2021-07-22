from typing import Optional
import numpy as np
from .functions import softmax, cross_entropy_error
from .types import NDArrayF


class MatMul:
    def __init__(self, W: NDArrayF) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x: Optional[NDArrayF] = None

    def forward(self, x: NDArrayF):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout: NDArrayF):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.params: list[NDArrayF] = []
        self.grads: list[NDArrayF] = []
        self.out: Optional[NDArrayF] = None

    def forward(self, x: NDArrayF):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: NDArrayF):
        dx = dout * (1.0 - self.out)*self.out
        return dx


class Affine:
    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x: Optional[NDArrayF] = None

    def forward(self, x: NDArrayF):
        W, b = self.params
        out = x @ W + b
        self.x = x
        return out

    def backward(self, dout: NDArrayF):
        W, _ = self.params
        dx = dout @ W.T
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.params[0][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Embedding:
    def __init__(self, W: NDArrayF) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class EmbeddingDot:
    def __init__(self, W: NDArrayF) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache: Optional[tuple] = None

    def forward(self, h: NDArrayF, idx: NDArrayF):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)

        return out

    def backward(self, dout: NDArrayF):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
