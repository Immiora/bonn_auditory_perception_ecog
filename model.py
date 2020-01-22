import chainer
from chainer import functions as F
from chainer import links as L

class CNN_T(chainer.ChainList):
    def __init__(self, n_out):
        links = []
        links.append(L.Convolution2D(1, 128, ksize=(19, 1), stride=(1, 1), pad = (9, 0), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(128))
        links.append(L.Convolution2D(128, 128, ksize=(15, 1), stride=(1, 1), pad = (7, 0), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(128))
        links.append(L.Convolution2D(128, 256, ksize=(9, 1), stride=(1, 1), pad = (4, 0), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(256))
        links.append(L.Convolution2D(256, 512, ksize=(5, 1), stride=(1, 1), pad = (2, 0), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(512))
        links.append(L.Convolution2D(512, n_out, ksize=(1, 1), stride=(1, 1), pad = (0, 0), initialW=chainer.initializers.GlorotUniform()))
        super(CNN_T, self).__init__(*links)

    def __call__(self, x, dur=1):
        x = F.pad(x, [(0, 0), (0, 0), (125 * dur, 125 * dur), (0, 0)], 'constant')
        z = F.relu(self[1](self[0](x)))
        z = F.dropout(F.max_pooling_2d(z, ksize=(15, 1), stride=(15, 1), pad=(0, 0)), .1)
        z = F.relu(self[3](self[2](z)))
        z = F.dropout(F.max_pooling_2d(z, ksize=(11, 1), stride=(11, 1), pad=(0, 0)), .1)
        z = F.relu(self[5](self[4](z)))
        z = F.relu(self[7](self[6](z)))
        z = self[8](z)
        z = F.squeeze(z)
        z = F.swapaxes(z, 1, 2)
        return z


class CNN_F(chainer.ChainList):
    def __init__(self, n_out):
        links = []
        links.append(L.Convolution2D(1, 64, ksize=(5, 7), stride=1, pad = (2, 3), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(64))
        links.append(L.Convolution2D(64, 128, ksize=(5, 7), stride=1, pad = (2, 3), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(128))
        links.append(L.Convolution2D(128, 256, ksize=(3, 7), stride=1, pad = (1, 3), initialW=chainer.initializers.GlorotUniform()))
        links.append(L.BatchNormalization(256))
        links.append(L.Convolution2D(256, n_out, ksize=(1, 1), stride=1, pad=(0, 0), initialW=chainer.initializers.GlorotUniform()))

        super(CNN_F, self).__init__(*links)

    def __call__(self, x):
        z = F.relu(self[1](self[0](x)))
        z = F.dropout(F.max_pooling_2d(z, ksize=(1, 5), stride=(1, 5), pad=(0, 0)), .1)
        z = F.relu(self[3](self[2](z)))
        z = F.dropout(F.max_pooling_2d(z, ksize=(1, 5), stride=(1, 5), pad=(0, 0)), .1)
        z = F.relu(self[5](self[4](z)))
        z = F.dropout(F.max_pooling_2d(z, ksize=(1, 7), stride=(1, 7), pad=(0, 0)), .1)
        z = self[6](z)
        z = F.squeeze(z)
        z = F.swapaxes(z, 1, 2)
        return z

class RNN(chainer.ChainList):
    def __init__(self, n_in, n_hid, n_out):
        links = []
        links.append(L.StatefulZoneoutLSTM(n_in, n_hid, c_ratio=.05, h_ratio=.05))
        links.append(L.Linear(None, n_out))
        self.n_out = n_out
        super(RNN, self).__init__(*links)

    def reset_state(self):
        self[0].reset_state()

    def __call__(self, x):
        y = []
        for tt in range(x.shape[1]):
            y.append(self[1](self[0](x[:, tt, :])))
        y = F.hstack(y)
        y = F.reshape(y, (x.shape[0], x.shape[1], self.n_out))
        return y

class RCNN(chainer.ChainList):
    def __init__(self, cnnt_out, cnnf_out, rnn_hid, rnn_out):
        links = []
        links.append(CNN_T(cnnt_out))
        links.append(CNN_F(cnnf_out))
        links.append(RNN(cnnt_out+cnnf_out, rnn_hid, rnn_out))
        super(RCNN, self).__init__(*links)

    def reset_state(self):
        self[-1].reset_state()

    def __call__(self, x, dur):
        z1 = self[0](x[0], dur)
        z2 = self[1](x[1])
        z = F.concat([z1, z2], 2)
        y = self[-1](z) # rnn output
        return y

