import numpy as np
import random
from random import sample
from keras.models import load_model
import copy


def binseq_to_burstseq(sequence):
    new_sequence = []
    start_point = 0
    sequence = np.ndarray.tolist(sequence)
    i = 1
    while i != len(sequence) - 1:
        if sequence[i]:
            if sequence[i] != sequence[i - 1]:
                new_sequence.append(len(sequence[start_point: i]) * sequence[i - 1])
                start_point = i
            i = i + 1
        elif not sequence[i]:
            new_sequence.append(len(sequence[start_point: i]) * sequence[i - 1])
            start_point = i
            i = len(sequence) - 1
    if len(new_sequence) < 750:
        new_sequence = new_sequence + ([0] * (750 - len(new_sequence)))
    elif len(new_sequence) > 750:
        new_sequence = new_sequence[:750]

    return np.array(new_sequence, dtype=np.int64)


def burstseq_to_binseq(sequence):
    new_sequence = []
    sequence = sequence.astype(np.int64)
    sequence = np.ndarray.tolist(sequence)
    for item in sequence[:len(sequence) - sequence.count(0)]:
        new_item = item / abs(item)
        new_sequence = new_sequence + ([new_item] * (abs(item)))

    if len(new_sequence) < 10000:
        new_sequence = new_sequence + ([0] * (10000 - len(new_sequence)))
    elif len(new_sequence) > 10000:
        new_sequence = new_sequence[:10000]
    return np.array(new_sequence, dtype=np.float64)


def make_pool(X, Y, class_num, size=200):
    indexes = np.where(Y != np.int64(class_num))
    # print(indexes[0][:1000])
    new_X = X[indexes]
    new_Y = Y[indexes]
    sampled = sample(range(len(new_Y)), size)
    pool = [{'x': new_X[i], 'y': new_Y[i]} for i in sampled]
    return pool


def select_trgt(pool, dic):
    pool.sort(key=lambda a: np.linalg.norm(a['x'] - dic['x']))
    return pool[0]


def compute_grad(I, It):
    D = np.linalg.norm(I - It)
    grad = (I - It) / D
    return grad


def compute_delta(I, It, alfa):
    grad = compute_grad(I, It)
    # print('gradient is : ', grad)
    delta = -1 * grad
    # print('grad_neg is : ', delta)
    delta[(delta * I) <= 0] = 0
    # print('delta is : ',  delta)
    delta = delta * alfa
    # print('delta * alfa is : ',  delta)
    return delta


def detector_deceived(I_manipulated, tau_c, detector):
    x = I_manipulated['x'].reshape(1, -1).astype('float32')[:, :, np.newaxis]
    result = detector.predict(x, verbose=2)
    if result[0][I_manipulated['y']] < tau_c:
        return True
    else:
        return False


def create_mockingbird(X, Y, I_src, detector, iter_num=500, tau_c=0.01, tau_D=0.0001, alfa=5):
    leaved_src_clss = False
    I_src_unchanged = copy.deepcopy(I_src)
    I_manipulated = copy.deepcopy(I_src)

    while leaved_src_clss is not True:

        ''' ToDo : 2 khat paeen : very very mashkook !! to maghaleh gofteh continiue ! yani
                ba hamin i_src dastkari shodeh ya ba I_src default?
                 next : what if delta that is made is float number how to round it ? !!'''
        I_src = copy.deepcopy(I_src_unchanged)
        pool = make_pool(X, Y, I_src['y'])
        I_trgt = select_trgt(pool, I_src)  # this line !
        counter = 0
        refill_counter = 0
        floated_I_src = binseq_to_burstseq(I_src['x']).astype(np.float32)

        while counter != iter_num and leaved_src_clss is not True and refill_counter < 10:
            delta = compute_delta(binseq_to_burstseq(I_src['x']), binseq_to_burstseq(I_trgt['x']), alfa)
            # I_src['x'] = burstseq_to_binseq(binseq_to_burstseq(I_src['x']) + np.rint(delta))
            # I_manipulated = I_src
            floated_I_src = floated_I_src + delta
            I_manipulated['x'] = burstseq_to_binseq(np.rint(floated_I_src))
            leaved_src_clss = detector_deceived(I_manipulated, tau_c, detector)
            if np.linalg.norm(delta) < tau_D:
                refill_counter = refill_counter + 1
            else:
                refill_counter = 0

            counter = counter + 1

    return I_manipulated
