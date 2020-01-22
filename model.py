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

# import chainer
# from chainer import functions as F
# from chainer import links as L
#
# class CNN_T(chainer.ChainList):
#     def __init__(self, n_out):
#         links = []
#         links.append(L.Convolution2D(1, 128, ksize=(21, 1), stride=(1, 1), pad = (10, 0), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(128))
#         links.append(L.Convolution2D(128, 128, ksize=(15, 1), stride=(1, 1), pad = (7, 0), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(128))
#         links.append(L.Convolution2D(128, 256, ksize=(7, 1), stride=(1, 1), pad = (3, 0), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(256))
#         links.append(L.Convolution2D(256, 512, ksize=(5, 1), stride=(1, 1), pad = (2, 0), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(512))
#         links.append(L.Convolution2D(512, 512, ksize=(5, 1), stride=(1, 1), pad = (2, 0), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(512))
#         links.append(L.Convolution2D(512, n_out, ksize=(1, 1), stride=(1, 1), pad = (0, 0), initialW=chainer.initializers.GlorotUniform()))
#         super(CNN_T, self).__init__(*links)
#
#     def __call__(self, x, dur=1):
#         x = F.pad(x, [(0, 0), (0, 0), (125 * dur, 125 * dur), (0, 0)], 'constant')
#         z = F.relu(self[1](self[0](x)))
#         z = F.max_pooling_2d(z, ksize=(19, 1), stride=(15, 1), pad=(0, 0))
#         z = F.relu(self[3](self[2](z)))
#         z = F.max_pooling_2d(z, ksize=(13, 1), stride=(11, 1), pad=(0, 0))
#         z = F.relu(self[5](self[4](z)))
#         z = F.relu(self[7](self[6](z)))
#         z = F.relu(self[9](self[8](z)))
#         z = self[10](z)
#         z = F.squeeze(z)
#         z = F.swapaxes(z, 1, 2)
#         return z
#
#
# class CNN_F(chainer.ChainList):
#     def __init__(self, n_out):
#         links = []
#         links.append(L.Convolution2D(1, 64, ksize=(5, 15), stride=1, pad = (2, 7), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(64))
#         links.append(L.Convolution2D(64, 128, ksize=(5, 11), stride=1, pad = (2, 5), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(128))
#         links.append(L.Convolution2D(128, 256, ksize=(3, 7), stride=1, pad = (1, 3), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(256))
#         links.append(L.Convolution2D(256, 256, ksize=(3, 5), stride=1, pad = (1, 2), initialW=chainer.initializers.GlorotUniform()))
#         links.append(L.BatchNormalization(256))
#         links.append(L.Convolution2D(256, n_out, ksize=(1, 1), stride=1, pad=(0, 0), initialW=chainer.initializers.GlorotUniform()))
#
#         super(CNN_F, self).__init__(*links)
#
#     def __call__(self, x):
#         x = F.pad(x, [(0, 0), (0, 0), (0, 0), (0, 5)], 'constant')
#         z = F.relu(self[1](self[0](x)))
#         z = F.max_pooling_2d(z, ksize=(1, 9), stride=(1, 7), pad=(0, 0))
#         z = F.relu(self[3](self[2](z)))
#         z = F.max_pooling_2d(z, ksize=(1, 7), stride=(1, 5), pad=(0, 0))
#         z = F.relu(self[5](self[4](z)))
#         z = F.relu(self[7](self[6](z)))
#         z = F.max_pooling_2d(z, ksize=(1, 7), stride=(1, 4), pad=(0, 0))
#         z = self[8](z)
#         z = F.squeeze(z)
#         z = F.swapaxes(z, 1, 2)
#         return z
#
# class RNN(chainer.ChainList):
#     def __init__(self, n_in, n_hid, n_out):
#         links = []
#         links.append(L.StatefulZoneoutLSTM(n_in, n_hid, c_ratio=.05, h_ratio=.05))
#         links.append(L.Linear(None, n_out))
#         self.n_out = n_out
#         super(RNN, self).__init__(*links)
#
#     def reset_state(self):
#         self[0].reset_state()
#
#     def __call__(self, x):
#         y = []
#         for tt in range(x.shape[1]):
#             y.append(self[1](self[0](x[:, tt, :])))
#         y = F.hstack(y)
#         y = F.reshape(y, (x.shape[0], x.shape[1], self.n_out))
#         return y
#
# class RCNN(chainer.ChainList):
#     def __init__(self, cnnt_out, cnnf_out, rnn_hid, rnn_out):
#         links = []
#         links.append(CNN_T(cnnt_out))
#         links.append(CNN_F(cnnf_out))
#         links.append(RNN(cnnt_out+cnnf_out, rnn_hid, rnn_out))
#         super(RCNN, self).__init__(*links)
#
#     def reset_state(self):
#         self[-1].reset_state()
#
#     def __call__(self, x, dur):
#         z1 = self[0](x[0], dur)
#         z2 = self[1](x[1])
#         z = F.concat([z1, z2], 2)
#         y = self[-1](z) # rnn output
#         return y
# # import chainer
# # from chainer import functions as F
# # from chainer import links as L
# #
# # class CNN_T(chainer.ChainList):
# #     def __init__(self, n_out):
# #         links = []
# #         links.append(L.Convolution2D(1, 128, ksize=(19, 1), stride=(1, 1), pad = (9, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 128, ksize=(15, 1), stride=(1, 1), pad = (7, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 256, ksize=(7, 1), stride=(1, 1), pad = (3, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(256))
# #         links.append(L.Convolution2D(256, 512, ksize=(5, 1), stride=(1, 1), pad = (2, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(512))
# #         links.append(L.Convolution2D(512, n_out, ksize=(1, 1), stride=(1, 1), pad = (0, 0), initialW=chainer.initializers.GlorotUniform()))
# #         super(CNN_T, self).__init__(*links)
# #
# #     def __call__(self, x, dur=1):
# #         x = F.pad(x, [(0, 0), (0, 0), (125 * dur, 125 * dur), (0, 0)], 'constant')
# #         z = F.relu(self[1](self[0](x)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(19, 1), stride=(15, 1), pad=(0, 0)), .1)
# #         z = F.relu(self[3](self[2](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(15, 1), stride=(11, 1), pad=(0, 0)), .1)
# #         z = F.relu(self[5](self[4](z)))
# #         z = F.relu(self[7](self[6](z)))
# #         z = self[8](z)
# #         z = F.squeeze(z)
# #         z = F.swapaxes(z, 1, 2)
# #         return z
# #
# #
# # class CNN_F(chainer.ChainList):
# #     def __init__(self, n_out):
# #         links = []
# #         links.append(L.Convolution2D(1, 64, ksize=(5, 11), stride=10, pad = (2, 5), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(64))
# #         links.append(L.Convolution2D(64, 128, ksize=(5, 5), stride=1, pad = (2, 2), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 256, ksize=(3, 3), stride=1, pad = (1, 1), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(256))
# #         links.append(L.Convolution2D(256, n_out, ksize=(1, 1), stride=1, pad=(0, 0), initialW=chainer.initializers.GlorotUniform()))
# #
# #         super(CNN_F, self).__init__(*links)
# #
# #     def __call__(self, x):
# #         x = F.pad(x, [(0, 0), (0, 0), (0, 0), (2, 2)], 'constant')
# #         z = F.relu(self[1](self[0](x)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 11), stride=(1, 9), pad=(0, 0)), .1)
# #         z = F.relu(self[3](self[2](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 5), stride=(1, 3), pad=(0, 0)), .1)
# #         z = F.relu(self[5](self[4](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 5), stride=(1, 3), pad=(0, 0)), .1)
# #         z = self[6](z)
# #         z = F.squeeze(z)
# #         z = F.swapaxes(z, 1, 2)
# #         return z
# #
# # class RNN(chainer.ChainList):
# #     def __init__(self, n_in, n_hid, n_out):
# #         links = []
# #         links.append(L.StatefulZoneoutLSTM(n_in, n_hid, c_ratio=.05, h_ratio=.05))
# #         links.append(L.Linear(None, n_out))
# #         self.n_out = n_out
# #         super(RNN, self).__init__(*links)
# #
# #     def reset_state(self):
# #         self[0].reset_state()
# #
# #     def __call__(self, x):
# #         y = []
# #         for tt in range(x.shape[1]):
# #             y.append(self[1](self[0](x[:, tt, :])))
# #         y = F.hstack(y)
# #         y = F.reshape(y, (x.shape[0], x.shape[1], self.n_out))
# #         return y
# #
# # class RCNN(chainer.ChainList):
# #     def __init__(self, cnnt_out, cnnf_out, rnn_hid, rnn_out):
# #         links = []
# #         links.append(CNN_T(cnnt_out))
# #         links.append(CNN_F(cnnf_out))
# #         links.append(RNN(cnnt_out+cnnf_out, rnn_hid, rnn_out))
# #         super(RCNN, self).__init__(*links)
# #
# #     def reset_state(self):
# #         self[-1].reset_state()
# #
# #     def __call__(self, x, dur):
# #         z1 = self[0](x[0], dur)
# #         z2 = self[1](x[1])
# #         z = F.concat([z1, z2], 2)
# #         y = self[-1](z) # rnn output
# #         return y
#
# # import chainer
# # from chainer import functions as F
# # from chainer import links as L
# # from chainer import initializers as I
# #
# # class CNN_T(chainer.ChainList):
# #     def __init__(self, n_out):
# #         links = []
# #         links.append(L.Convolution2D(1, 128, ksize=(15, 1), stride=(1, 1), pad = (7, 0), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 128, ksize=(11, 1), stride=(1, 1), pad = (5, 0), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 256, ksize=(7, 1), stride=(1, 1), pad = (3, 0), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(256))
# #         links.append(L.Convolution2D(256, 512, ksize=(5, 1), stride=(1, 1), pad = (2, 0), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(512))
# #         links.append(L.Convolution2D(512, n_out, ksize=(1, 1), stride=(1, 1), pad = (0, 0), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         super(CNN_T, self).__init__(*links)
# #
# #     def __call__(self, x, dur=1):
# #         x = F.pad(x, [(0, 0), (0, 0), (125 * dur, 125 * dur), (0, 0)], 'constant')
# #         z = F.relu(self[1](self[0](x)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(15, 1), stride=(15, 1), pad=(0, 0)), .1)
# #         z = F.relu(self[3](self[2](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(11, 1), stride=(11, 1), pad=(0, 0)), .1)
# #         z = F.relu(self[5](self[4](z)))
# #         z = F.relu(self[7](self[6](z)))
# #         z = self[8](z)
# #         z = F.squeeze(z)
# #         z = F.swapaxes(z, 1, 2)
# #         return z
# #
# # ##
# # class CNN_F(chainer.ChainList):
# #     def __init__(self, n_out):
# #         links = []
# #         links.append(L.Convolution2D(1, 128, ksize=(1, 35), stride=(1, 1), pad = (0, 7), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 128, ksize=(1, 21), stride=(1, 1), pad = (0, 5), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 256, ksize=(1, 11), stride=(1, 1), pad = (0, 3), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         links.append(L.BatchNormalization(256))
# #         links.append(L.Convolution2D(256, n_out, ksize=(1, 1), stride=(1, 1), pad = (0, 0), initialW=I.GlorotUniform(), initial_bias=I.Zero()))
# #         super(CNN_F, self).__init__(*links)
# #
# #     def __call__(self, x, dur=1, ntile=50):
# #         x = F.pad(x, [(0, 0), (0, 0), (0, 0), (7 * dur/x.shape[2], 8 * dur/x.shape[2])], 'constant')
# #         z = F.relu(self[1](self[0](x)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 35), stride=(1, 35), pad=(0, 0)), .1)
# #         z = F.relu(self[3](self[2](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 21), stride=(1, 21), pad=(0, 1)), .1)
# #         z = F.relu(self[5](self[4](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 11), stride=(1, 11), pad=(0, 0)), .1)
# #         z = self[6](z)
# #         z = F.tile(z, (1, 1, 1, ntile))
# #         z = F.reshape(z, (z.shape[0], z.shape[1], -1))
# #         z = F.swapaxes(z, 1, 2)
# #         return z
# #
# # class RNN(chainer.ChainList):
# #     def __init__(self, n_in, n_hid, n_out):
# #         links = []
# #         links.append(L.StatefulZoneoutLSTM(n_in, n_hid, c_ratio=.05, h_ratio=.05))
# #         links.append(L.Linear(None, n_out))
# #         self.n_out = n_out
# #         super(RNN, self).__init__(*links)
# #
# #     def reset_state(self):
# #         self[0].reset_state()
# #
# #     def __call__(self, x):
# #         y = []
# #         for tt in range(x.shape[1]):
# #             y.append(self[1](self[0](x[:, tt, :])))
# #         y = F.hstack(y)
# #         y = F.reshape(y, (x.shape[0], x.shape[1], self.n_out))
# #         return y
# #
# # class RCNN(chainer.ChainList):
# #     def __init__(self, cnnt_out, cnnf_out, rnn_hid, rnn_out):
# #         links = []
# #         links.append(CNN_T(cnnt_out))
# #         links.append(CNN_F(cnnf_out))
# #         links.append(RNN(cnnt_out+cnnf_out, rnn_hid, rnn_out))
# #         super(RCNN, self).__init__(*links)
# #
# #     def reset_state(self):
# #         self[-1].reset_state()
# #
# #     def __call__(self, x, dur):
# #         z1 = self[0](x[0], dur)
# #         z2 = self[1](x[1], dur)
# #         z = F.concat([z1, z2], 2)
# #         y = self[-1](z) # rnn output
# #         return y
#
# # import chainer
# # from chainer import functions as F
# # from chainer import links as L
# #
# # class CNN_T(chainer.ChainList):
# #     def __init__(self, n_out):
# #         links = []
# #         links.append(L.Convolution2D(1, 128, ksize=(25, 1), stride=(1, 1), pad = (12, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 256, ksize=(19, 1), stride=(1, 1), pad = (9, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(256))
# #         links.append(L.Convolution2D(256, 512, ksize=(11, 1), stride=(1, 1), pad = (5, 0), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(512))
# #         links.append(L.Convolution2D(512, n_out, ksize=(1, 1), stride=(1, 1), pad = (0, 0), initialW=chainer.initializers.GlorotUniform()))
# #         super(CNN_T, self).__init__(*links)
# #
# #     def __call__(self, x, dur=1):
# #         x = F.pad(x, [(0, 0), (0, 0), (125 * dur, 125 * dur), (0, 0)], 'constant')
# #         z = F.relu(self[1](self[0](x)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(19, 1), stride=(15, 1), pad=(0, 0)), .1)
# #         z = F.relu(self[3](self[2](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(15, 1), stride=(11, 1), pad=(0, 0)), .1)
# #         z = F.relu(self[5](self[4](z)))
# #         z = self[6](z)
# #         z = F.squeeze(z)
# #         z = F.swapaxes(z, 1, 2)
# #         return z
# #
# #
# # class CNN_F(chainer.ChainList):
# #     def __init__(self, n_out):
# #         links = []
# #         links.append(L.Convolution2D(1, 128, ksize=(5, 19), stride=1, pad = (2, 9), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(128))
# #         links.append(L.Convolution2D(128, 256, ksize=(5, 7), stride=1, pad = (2, 2), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(256))
# #         links.append(L.Convolution2D(256, 512, ksize=(3, 5), stride=1, pad = (1, 2), initialW=chainer.initializers.GlorotUniform()))
# #         links.append(L.BatchNormalization(512))
# #         links.append(L.Convolution2D(512, n_out, ksize=(1, 1), stride=1, pad=(0, 0), initialW=chainer.initializers.GlorotUniform()))
# #
# #         super(CNN_F, self).__init__(*links)
# #
# #     def __call__(self, x):
# #         x = F.pad(x, [(0, 0), (0, 0), (0, 0), (3, 4)], 'constant')
# #         z = F.relu(self[1](self[0](x)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 17), stride=(1, 15), pad=(0, 0)), .1)
# #         z = F.relu(self[3](self[2](z)))
# #         z = F.dropout(F.max_pooling_2d(z, ksize=(1, 11), stride=(1, 9), pad=(0, 0)), .1)
# #         z = F.relu(self[5](self[4](z)))
# #         z = self[6](z)
# #         z = F.squeeze(z)
# #         z = F.swapaxes(z, 1, 2)
# #         return z
# #
# # class RNN(chainer.ChainList):
# #     def __init__(self, n_in, n_hid, n_out):
# #         links = []
# #         links.append(L.StatefulZoneoutLSTM(n_in, n_hid, c_ratio=.05, h_ratio=.05))
# #         links.append(L.Linear(None, n_out))
# #         self.n_out = n_out
# #         super(RNN, self).__init__(*links)
# #
# #     def reset_state(self):
# #         self[0].reset_state()
# #
# #     def __call__(self, x):
# #         y = []
# #         for tt in range(x.shape[1]):
# #             y.append(self[1](self[0](x[:, tt, :])))
# #         y = F.hstack(y)
# #         y = F.reshape(y, (x.shape[0], x.shape[1], self.n_out))
# #         return y
# #
# # class RCNN(chainer.ChainList):
# #     def __init__(self, cnnt_out, cnnf_out, rnn_hid, rnn_out):
# #         links = []
# #         links.append(CNN_T(cnnt_out))
# #         links.append(CNN_F(cnnf_out))
# #         links.append(RNN(cnnt_out+cnnf_out, rnn_hid, rnn_out))
# #         super(RCNN, self).__init__(*links)
# #
# #     def reset_state(self):
# #         self[-1].reset_state()
# #
# #     def __call__(self, x, dur):
# #         z1 = self[0](x[0], dur)
# #         z2 = self[1](x[1])
# #         z = F.concat([z1, z2], 2)
# #         y = self[-1](z) # rnn output
# #         return y