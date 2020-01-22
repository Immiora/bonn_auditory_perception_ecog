import numpy as np
import os
import psutil
import librosa
from fractions import Fraction
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import OrderedDict
from scipy.stats import rankdata
from model import RCNN
import chainer
from chainer import functions as F
from chainer import serializers


class Scaler():
    def __init__(self):
        self.scalers = OrderedDict()

    def __call__(self, ktrain, ktest=None, kval=None):
        for s in range(len(ktrain)):
            s_scaler = StandardScaler()
            ktrain[s] = s_scaler.fit_transform(ktrain[s])
            if ktest is not None: ktest[s] = s_scaler.transform(ktest[s])
            if kval is not None: kval[s] = s_scaler.transform(kval[s])
            self.scalers[s] = {'mean': s_scaler.mean_, 'std': s_scaler.scale_}
        return ktrain, ktest, kval

		
def reshape3(x, dim2):
    return x.reshape(x.shape[0]/dim2, dim2, x.shape[-1])

def split_list(l, wanted_parts=1):
    length = len(l)
    return [ l[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]	

def resample(x, sr1=25, sr2=125, axis=0):
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)
	
def roll_data(xs, shifts):
    if type(xs) is list:
        assert(type(shifts) is list)
        rolled = [np.roll(x, -int(shift), 0) for (x, shift) in zip(xs, shifts)]
    else:
        rolled = np.roll(xs, -int(shifts), 0)
    return rolled

def prepare_input(xs, srs, n_back):
    # pad with zeros and reshape to 3d with n_back for rnn as dim 1
	
    def prepare_one(x, sr):
        x_pad = n_back * sr - (x.shape[0] % (n_back * sr))
        return reshape3(np.pad(x, [(0, x_pad), (0, 0)], 'constant'), n_back * sr)

    if type(xs) is list:
        assert(type(srs) is list)
        xs = [prepare_one(x, sr) for x, sr in zip(xs, srs)]
    else:
        xs = prepare_one(xs, srs)
    return xs

def get_batches(xs, batch_size):
    # dur-long sample in continuous order of data

    assert([x.shape[0] for x in xs][1:] == [x.shape[0] for x in xs][:-1]) # same shape[0] for all in xs
    len_seq = xs[0].shape[0]
    indices = range(len_seq)

    if len_seq % batch_size != 0: batch_size = len_seq / (len_seq // batch_size) + 1
    batch_indices = split_list(indices, batch_size)

    n_batches = min([len(i) for i in batch_indices])  # any way to keep this? reset_state is needed

    bs = [[] for _ in range(len(xs))]
    for i_batch in range(n_batches):
        b = [i[i_batch] for i in batch_indices if i_batch < len(i)]
        [bs[xi].append(x[b]) for xi, x in enumerate(xs)]

    return bs

def acc_pass(t_, y_):
	# spearman correlation by default, can be adjusted to any desired similarity metric
    r = np.array(map(lambda x, y: np.corrcoef(rankdata(x), rankdata(y))[0, 1], t_.T, y_.T))
    return r

##
def main():
    #
    print('\nRunnig fold: ' + sys.argv[1])
    kfold_ = int(sys.argv[1]) # only train for one cross-validation fold at a time (this way we can train all folds in parallel)
    print(type(kfold_))

    # load data
    tr_fact = 1 # 1 is 100% data for training

    out_dir = './results/rcnn_merge_time_coch_cval10_brain_hfb/' + \
              'n_back_6_cnnT_300h_cnnF_100h_rnn_300h_alt_alt2_concattest_train' + str(int(tr_fact * 100)) + '/'
    x1_file = './data/M3_audio_mono_down.wav'
    x2_file = './data/minoes_wav_freq_125Hz_abs.npy'
    t_file  = './data/minoes_hfb_6subjs_noduiven.npy'
    xtr1    = librosa.load(x1_file, sr=8000)[0]
    xtr2    = np.load(x2_file).astype(np.float32)
    ttr     = np.load(t_file).astype(np.float32)
    print('Train data: ' + str(int(tr_fact * 100)) + '%')

    # resample brain and spectrogram data to 50 Hz
    xtr2    = resample(xtr2, sr1=50, sr2=125)
    ttr     = resample(ttr, sr1=50, sr2=125)

    # take a sample in sec
    global sr1, sr2, sr3, n_back
    sr1     = 8000
    sr2     = 50
    sr3     = 50
    nsec    = ttr.shape[0] / float(sr2)
    nsamp   = nsec * 1
    n2      = int(nsamp * sr2)
    n3      = int(nsamp * sr3)
    xtr2    = xtr2[:n2]
    ttr     = ttr[:n3]

    # cut raw audio to match brain data (ttr) length in sec
    n1      = int(nsamp * sr1)
    xtr1    = xtr1[:n1]
    xtr1    = xtr1[:, None]

    # set up cross-validation for performance accuracy: set-up the same way for all folds when folds are trained separately
    kfolds = 10
    nparts = 7 # test set is not a continuous chunk but is a concatenation of nparts fragments for better performance
    ind1 = np.arange(xtr1.shape[0])
    ind2 = np.arange(ttr.shape[0])
    ind3 = np.arange(ttr.shape[0])
    TestI_, TestI = [], []
    kf = KFold(n_splits=kfolds * nparts)

    for (_, ix1_test), (_, ix2_test), (_, it_test) in zip(kf.split(xtr1), kf.split(xtr2), kf.split(ttr)):
        TestI_.append([ix1_test, ix2_test, it_test])

    for kfold in range(kfolds):
        TestI.append([np.array(
            [item for sublist in [TestI_[i][j] for i in range(0 + kfold, kfolds * nparts + kfold, kfolds)] for item in
             sublist])
                      for j in range(len(TestI_[0]))])


    if (out_dir is not None) & (not os.path.exists(out_dir)): os.makedirs(out_dir)
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024 / 1024 / 1024)

    # standard sklearn preprocessing of data
    scaler = Scaler()
    kfold = kfold_
    ktrain, ktest, _ = scaler([xtr1[np.setdiff1d(ind1, TestI[kfold][0])], xtr2[np.setdiff1d(ind2, TestI[kfold][1])], ttr[np.setdiff1d(ind3, TestI[kfold][2])]],
                              [xtr1[TestI[kfold][0]], xtr2[TestI[kfold][1]], ttr[TestI[kfold][2]]], None)

    nsec_tr    = ktrain[-1].shape[0] / float(sr2)
    nsamp_tr   = nsec_tr * tr_fact
    ktrain = map(lambda x, n: x.copy()[:n], ktrain, [int(nsamp_tr *i) for i in [sr1, sr2, sr3]])
    print(map(len, ktrain))
    print(map(len, ktest))

    # model parameters
    dur     = 1 # sec units
    batch_size = 16
    n_back  = 6 * dur # in dur units, temporal window of input data (how much data the model sees at once)
    nepochs = 30
    n_out   = ttr.shape[-1]
    alpha   = 5e-04
    h_cnn_t   = 300 # number of hidden units on top layer of CNN time
    h_cnn_f   = 100 # number of hidden units on top layer of CNN freq/spectra
    h_rnn   = 300 # number of hidden units of RNN

    print('batch size: ' + str(batch_size) + ', nepochs: ' + str(nepochs) + ', lr: ' + str(alpha) +
                            ', h_cnn_t: ' + str(h_cnn_t) + ', h_cnn_f: ' + str(h_cnn_f) + ', h_rnn: ' + str(h_rnn))
    print('outdir: ' + out_dir)

    # set up model
    rcnn = RCNN(h_cnn_t, h_cnn_f, h_rnn, n_out)
    opt = chainer.optimizers.Adam(alpha)
    opt.setup(rcnn)
	
    with open(out_dir + 'fold' + str(kfold) + '_run.log', 'wb'): pass # running epoch and best performance are saved to txt file for bookkeeping
    with open(out_dir + 'fold' + str(kfold) + '_epoch.txt', 'wb'): pass

    # train loop
    best_acc = -1
    for epoch in range(nepochs):
        print('Epoch ' + str(epoch))
        with open(out_dir + 'fold' + str(kfold) + '_run.log', 'a') as fid0:
            fid0.write('epoch' + str(epoch) + '\n')
        rcnn.reset_state()
        x1, x2, t = roll_data(ktrain, [.14 * epoch * sr for sr in [sr1, sr2, sr3]])
        x1, x2, t = prepare_input([x1, x2, t], [sr1, sr2, sr3], n_back)
        xbs1, xbs2, tbs = get_batches([x1, x2, t], batch_size)
        print(process.memory_info().rss / 1024 / 1024 / 1024)

        for ib, (xb1, xb2, tb) in enumerate(zip(xbs1, xbs2, tbs)):
            with chainer.using_config('train', True):
                y = rcnn([np.expand_dims(xb1, 1), np.expand_dims(xb2, 1)], n_back)
                loss = 0
                for ni in range(y.shape[1]):
                    loss += F.mean_squared_error(tb[:, ni, :], y[:, ni, :])
                r = acc_pass(tb.reshape((-1, n_out)), y.data.reshape((-1, n_out)))
                print('\t\tbatch ' + str(ib) + ', train loss: ' + str(loss.data / tb.shape[1]) + ', max acc: ' + str(np.max(r)))
                rcnn.cleargrads()
                loss.backward()
                loss.unchain_backward()
                opt.update()

        xb1_, xb2_, tb_ = prepare_input(ktest, [sr1, sr2, sr3], n_back)
        rcnn.reset_state()
        with chainer.using_config('train', False):
            y_ = rcnn([np.expand_dims(xb1_, 1), np.expand_dims(xb2_, 1)], n_back)
            loss_ = 0
            for ni in range(y_.shape[1]):
                loss_ += F.mean_squared_error(tb_[:, ni, :], y_[:, ni, :])

        r = acc_pass(tb_.reshape((-1, n_out)), y_.data.reshape((-1, n_out)))
        print('\t\ttest loss: ' + str(np.round(loss_.data / tb_.shape[1], 3)) + ', max acc: ' + str(
            np.round(np.sort(r)[::-1][:10], 4)))
        run_acc = np.mean(np.sort(r)[::-1][:10])
        if run_acc > best_acc: # only if performance of current model is superior, save it to file
            print('Current model is best: ' + str(np.round(run_acc, 4)) + ' > ' + str(
                np.round(best_acc, 4)) + ': saving update to disk')
            best_acc = run_acc.copy()
            serializers.save_npz(out_dir + '/model' + str(kfold) + '.npz', rcnn)
            with open(out_dir + 'fold' + str(kfold) + '_epoch.txt', 'a') as fid:
                fid.write(str(epoch) + '\n')
                fid.write(str(np.sort(r)[::-1][:10]) + '\n')
            np.save(out_dir + '/predictions_fold' + str(kfold), y_.data.reshape((-1, n_out)))
            np.save(out_dir + '/targets_fold' + str(kfold), tb_.reshape((-1, n_out)))

##
if __name__ == '__main__':
    main()
