from typing import Optional
import numpy as np
from .functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W: np.ndarray) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout: np.ndarray):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x: np.ndarray):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray):
        dx = dout * (1.0 - self.out)*self.out
        return dx


class Affine:
    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x: np.ndarray):
        W, b = self.params
        out = x @ W + b
        self.x = x
        return out

    def backward(self, dout: np.ndarray):
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
    def __init__(self, W: np.ndarray) -> None:
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


class EmbeddingDot:
    def __init__(self, W: np.ndarray) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h: np.ndarray, idx: np.ndarray):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)

        return out

    def backward(self, dout: np.ndarray):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dot * target_W
        return dh
