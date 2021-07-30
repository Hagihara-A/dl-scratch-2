from common.trainer import RnnlmTrainer
from common.np import GPU
from dataset import ptb
from common.util import eval_perplexity, to_gpu
from common.base_model import BaseModel
from common.np import np
from common.optimizers import SGD
import common.time_layers as TL


class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        normal = np.random.normal
        mu = 0
        root = np.sqrt

        embed_W = normal(mu, 0.01, (V, D), dtype=np.float16)
        lstm_Wx1 = normal(mu, 1/root(D), (D, 4*H), dtype=np.float16)
        lstm_Wh1 = normal(mu, 1/root(H), (H, 4*H), dtype=np.float16)
        lstm_b1 = np.zeros(4*H, dtype=np.float16)
        lstm_Wx2 = normal(mu, 1/root(D), (H, 4*H), dtype=np.float16)
        lstm_Wh2 = normal(mu, 1/root(H), (H, 4*H), dtype=np.float16)
        lstm_b2 = np.zeros(4*H,  dtype=np.float16)
        affine_b = np.zeros(V, dtype=np.float16)

        self.layers = (
            TL.TimeEmbedding(embed_W),
            TL.TimeDropout(dropout_ratio),
            TL.TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TL.TimeDropout(dropout_ratio),
            TL.TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TL.TimeDropout(dropout_ratio),
            TL.TimeAffine(embed_W.T, affine_b)
        )
        self.loss_layer = TL.TimeSoftmaxWithLoss()
        self.lstm_layers = (self.layers[2], self.layers[4])
        self.drop_layers = (self.layers[1], self.layers[3], self.layers[5])
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def predict(self, xs, ts, train_flg=True):
        for layer in self.layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, ts, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()

            from common.optimizers import SGD


# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')

if GPU:
    corpus = to_gpu(corpus)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(vocab_size, wordvec_size,
                    hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                time_size=time_size, max_grad=max_grad)

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('valid perplexity: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)


# テストデータでの評価
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)
