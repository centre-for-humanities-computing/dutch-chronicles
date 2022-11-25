"""
Adaptive Fluctuation Analysis


Usage example
-------------

```
import numpy as np
import matplotlib.pyplot as plt

pure = np.linspace(-1, 1, 100)
noise = np.random.normal(0, 1, 100)
signal = pure + noise

example_weak_smooth = adaptive_filter(signal, span=56)
example_strong_smooth = adaptive_filter(signal, span=28)

plt.plot(signal, alpha=0.5, c='grey')
plt.plot(example_weak_smooth, c='blue')
plt.plot(example_strong_smooth, c='red')

```

Source
------
Minimal refactoring of old code from the Centre for Humanities Computing
https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/saffine/detrending_coeff.py
https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/saffine/detrending_method.py
https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/saffine/multi_detrending.py
"""

import numpy as np


def normalize(x, lower=-1, upper=1):
    """ transform x to x_ab in range [a, b]
    """
    x_norm = (upper - lower)*((x - np.min(x)) / (np.max(x) - np.min(x))) + lower
    return x_norm


def detrending_coeff(win_len, order):

    n = (win_len-1)/2
    A = np.mat(np.ones((win_len, order+1)))
    x = np.arange(-n, n+1)
    for j in range(0, order + 1):
        A[:, j] = np.mat(x ** j).T

    coeff_output = (A.T * A).I * A.T
    return coeff_output, A


def detrending_method(data, seg_len, fit_order):

    nrows, ncols = data.shape
    if nrows < ncols:
        data = data.T

    # seg_len = 1001,odd number
    nonoverlap_len = int((seg_len - 1) / 2)
    # get nrows again, in case data was transposed
    data_len = data.shape[0]
    # calculate the coefficient,given a window size and fitting order
    coeff_output, A = detrending_coeff(seg_len, fit_order)
    A_coeff = A * coeff_output

    for seg_index in range(1, 2):
        # left trend

        xi = np.arange(1 + (seg_index - 1) * (seg_len - 1),
                       seg_index * (seg_len - 1) + 2)
        xi_left = np.mat(xi)
        xi_max = xi.max()
        xi_min = xi.min()
        seg_data = data[xi_min - 1: xi_max, 0]
        left_trend = (A_coeff * seg_data).T

        # mid trend

        if seg_index * (seg_len - 1) + 1 + nonoverlap_len > data_len:
            xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) +
                           nonoverlap_len, data_len + 1)
            xi_mid = np.mat(xi)
            xi_max = xi.max()
            xi_min = xi.min()
            seg_data = data[xi_min - 1: xi_max, 0]
            nrows_seg = seg_data.shape[0]

            if nrows_seg < seg_len:
                coeff_output1, A1 = detrending_coeff(nrows_seg, fit_order)
                A_coeff1 = A1 * coeff_output1
                mid_trend = (A_coeff1 * seg_data).T
            else:
                mid_trend = (A_coeff * seg_data).T

            xx1 = left_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
            xx2 = mid_trend[0, 0: int((seg_len + 1) / 2)]
            w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
            xx_left = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

            record_x = xi_left[0, 0: nonoverlap_len]
            record_y = left_trend[0, 0: nonoverlap_len]
            mid_start_index = np.mat([(j) for j in range(np.shape(
                xi_mid)[1]) if xi_mid[0, j] == xi_left[0, np.shape(xi_left)[1] - 1] + 1])
            nrows_mid = mid_start_index.shape[0]
            mid_start_index = mid_start_index[0, 0]

            if nrows_mid == 0:
                record_x = np.hstack((record_x, xi_left[0, int(
                    (np.shape(xi_left)[1] + 3) / 2)-1: np.shape(xi_left)[1]]))
                record_y = np.hstack(
                    (record_y, xx_left[0, 1: np.shape(xx_left)[1]]))
            else:
                record_x = np.hstack((record_x, xi_left[0, int((np.shape(xi_left)[
                                     1] + 1) / 2)-1: np.shape(xi_left)[1]], xi_mid[0, mid_start_index: np.shape(xi_mid)[1]]))
                record_y = np.hstack((record_y, xx_left[0: np.shape(xx_left)[
                                     1]], mid_trend[0, int((seg_len + 3) / 2) - 1: np.shape(mid_trend)[1]]))

            detrended_data = data - record_y.T

            return detrended_data, record_y

        else:
            xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) +
                           nonoverlap_len, seg_index * (seg_len - 1) + nonoverlap_len + 2)
            xi_mid = np.mat(xi)
            xi_max = xi.max()
            xi_min = xi.min()
            seg_data = data[xi_min-1: xi_max, 0]
            nrows_seg = seg_data.shape[0]
            mid_trend = (A_coeff * seg_data).T

        # right trend

            if (seg_index + 1) * (seg_len - 1) + 1 > data_len:
                xi = np.arange(seg_index * (seg_len - 1) + 1, data_len + 1)
                xi_right = np.mat(xi)
                xi_max = xi.max()
                xi_min = xi.min()
                seg_data = data[xi_min - 1: xi_max, 0]
                nrows_seg = seg_data.shape[0]

                if nrows_seg < seg_len:
                    coeff_output1, A1 = detrending_coeff(nrows_seg, fit_order)
                    A_coeff1 = A1 * coeff_output1
                    right_trend = (A_coeff1 * seg_data).T
                else:
                    right_trend = (A_coeff * seg_data).T

                xx1 = left_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
                xx2 = mid_trend[0, 0: int((seg_len + 1) / 2)]
                w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
                xx_left = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

                xx1 = mid_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
                xx2 = right_trend[0, 0: int((seg_len + 1) / 2)]
                w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
                xx_right = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

                record_x = xi_left[0, 0: nonoverlap_len]
                record_y = left_trend[0, 0: nonoverlap_len]

                record_x = np.hstack((record_x, xi_left[0, int((np.shape(xi_left)[1] + 1) / 2) - 1: np.shape(
                    xi_left)[1]], xi_mid[0, int((np.shape(xi_mid)[1] + 1) / 2): np.shape(xi_mid)[1]]))
                record_y = np.hstack((record_y, xx_left[0, 0: np.shape(xx_left)[
                                     1]], xx_right[0, 1: np.shape(xx_right)[1]]))

                right_start_index = np.mat([(j) for j in range(np.shape(
                    xi_right)[1]) if xi_right[0, j] == xi_mid[0, np.shape(xi_mid)[1] - 1] + 1])
                right_start_index = right_start_index[0, 0]
                record_x = np.hstack(
                    (record_x, xi_right[0, right_start_index: np.shape(xi_right)[1]]))
                record_y = np.hstack(
                    (record_y, right_trend[0, right_start_index: np.shape(right_trend)[1]]))
                detrended_data = data - record_y.T

                return detrended_data, record_y

            else:
                xi = np.arange(seg_index * (seg_len - 1) + 1,
                               (seg_index + 1) * (seg_len - 1) + 2)
                xi_right = np.mat(xi)
                xi_max = xi.max()
                xi_min = xi.min()
                seg_data = data[xi_min - 1: xi_max, 0]
                right_trend = (A * coeff_output * seg_data).T

                xx1 = left_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
                xx2 = mid_trend[0, 0: int((seg_len + 1) / 2)]
                w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
                xx_left = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

                xx1 = mid_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
                xx2 = right_trend[0, 0: int((seg_len + 1) / 2)]
                w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
                xx_right = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

                record_x = xi_left[0, 0: nonoverlap_len]
                record_y = left_trend[0, 0: nonoverlap_len]

                record_x = np.hstack((record_x, xi_left[0, int((np.shape(xi_left)[1] + 1) / 2) - 1: np.shape(
                    xi_left)[1]], xi_mid[0, int((np.shape(xi_mid)[1] + 1) / 2): np.shape(xi_mid)[1]]))
                record_y = np.hstack((record_y, xx_left[0, 0: np.shape(xx_left)[
                                     1]], xx_right[0, 1: np.shape(xx_right)[1]]))

    for seg_index in range(2, int((data_len - 1) / (seg_len - 1))):
        # left_trend
        xi = np.arange((seg_index - 1) * (seg_len - 1) +
                       1, seg_index * (seg_len - 1) + 2)
        xi_left = np.mat(xi)
        xi_max = xi.max()
        xi_min = xi.min()
        seg_data = data[xi_min - 1: xi_max, 0]
        left_trend = (A_coeff * seg_data).T

        # mid trend

        xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) +
                       nonoverlap_len, seg_index * (seg_len - 1) + nonoverlap_len + 2)
        xi_mid = np.mat(xi)
        xi_max = xi.max()
        xi_min = xi.min()
        seg_data = data[xi_min - 1: xi_max, 0]
        mid_trend = (A_coeff * seg_data).T

        # right trend

        xi = np.arange(seg_index * (seg_len - 1) + 1,
                       (seg_index + 1) * (seg_len - 1) + 2)
        xi_right = np.mat(xi)
        xi_max = xi.max()
        xi_min = xi.min()
        seg_data = data[xi_min - 1: xi_max, 0]
        right_trend = (A_coeff * seg_data).T

        xx1 = left_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
        xx2 = mid_trend[0, 0: int((seg_len + 1) / 2)]
        w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
        xx_left = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

        xx1 = mid_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
        xx2 = right_trend[0, 0: int((seg_len + 1) / 2)]
        w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
        xx_right = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

        record_x = np.hstack((record_x, xi_left[0, int((np.shape(xi_left)[1] + 3) / 2) - 1: np.shape(
            xi_left)[1]], xi_mid[0, int((np.shape(xi_mid)[1] + 1) / 2): np.shape(xi_mid)[1]]))
        record_y = np.hstack((record_y, xx_left[0, 1: np.shape(xx_left)[
                             1]], xx_right[0, 1: np.shape(xx_right)[1]]))

    # last part of data

    for seg_index in range(int((data_len - 1) / (seg_len - 1)), int((data_len - 1) / (seg_len - 1)) + 1):
        # left trend

        xi = np.arange((seg_index - 1) * (seg_len - 1) +
                       1, seg_index * (seg_len - 1) + 2)
        xi_left = np.mat(xi)
        xi_max = xi.max()
        xi_min = xi.min()
        seg_data = data[xi_min - 1: xi_max, 0]
        left_trend = (A_coeff * seg_data).T

        # mid trend

        if seg_index * (seg_len - 1) + 1 + nonoverlap_len > data_len:
            xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) +
                           nonoverlap_len, data_len + 1)
            xi_mid = np.mat(xi)
            xi_max = xi.max()
            xi_min = xi.min()
            seg_data = data[xi_min - 1: xi_max, 0]
            nrows_seg = np.shape(seg_data)[0]

            if nrows_seg < seg_len:
                coeff_output1, A1 = detrending_coeff(nrows_seg, fit_order)
                A_coeff1 = A1 * coeff_output1
                mid_trend = (A_coeff1 * seg_data).T
            else:
                mid_trend = (A_coeff * seg_data).T

            xx1 = left_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
            xx2 = mid_trend[0, 0: int((seg_len + 1) / 2)]
            w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
            xx_left = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)
            mid_start_index = np.mat([(j) for j in range(np.shape(
                xi_mid)[1]) if xi_mid[0, j] == xi_left[0, np.shape(xi_left)[1] - 1] + 1])
            nrows_mid = np.shape(mid_start_index)[0]
            mid_start_index = mid_start_index[0, 0]

            if nrows_mid == 0:

                record_x = np.hstack((record_x, xi_left[0, int(
                    (np.shape(xi_left)[1] + 3) / 2) - 1: np.shape(xi_left)[1]]))
                record_y = np.hstack(
                    (record_y, xx_left[0, 1: np.shape(xx_left)[1]]))

            else:
                record_x = np.hstack((record_x, xi_left[0, int((np.shape(xi_left)[
                                     1] + 3) / 2) - 1: np.shape(xi_left)[1]], xi_mid[0, mid_start_index: np.shape(xi_mid)[1]]))
                record_y = np.hstack((record_y, xx_left[0, 1: np.shape(xx_left)[
                                     1]], mid_trend[0, int((seg_len + 3) / 2) - 1: np.shape(mid_trend)[1]]))

            detrended_data = data - record_y.T

            return detrended_data, record_y

        else:
            xi = np.arange(1 + (seg_index - 1) * (seg_len - 1) +
                           nonoverlap_len, seg_index * (seg_len - 1) + nonoverlap_len + 2)
            xi_mid = np.mat(xi)
            xi_max = xi.max()
            xi_min = xi.min()
            seg_data = data[xi_min - 1: xi_max, 0]
            mid_trend = (A_coeff * seg_data).T

        # right trend
        xi = np.arange(seg_index * (seg_len - 1) + 1, data_len + 1)
        xi_right = np.mat(xi)
        xi_max = xi.max()
        xi_min = xi.min()
        seg_data = data[xi_min - 1: xi_max, 0]
        nrows_seg = np.shape(seg_data)[0]

        if nrows_seg < seg_len:
            coeff_output1, A1 = detrending_coeff(nrows_seg, fit_order)
            A_coeff1 = A1 * coeff_output1
            right_trend = (A_coeff1 * seg_data).T
        else:
            right_trend = (A_coeff * seg_data).T

        xx1 = left_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
        xx2 = mid_trend[0, 0: int((seg_len + 1) / 2)]
        w = np.arange(0, nonoverlap_len + 1)/nonoverlap_len
        xx_left = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

        xx1 = mid_trend[0, int((seg_len + 1) / 2) - 1: seg_len]
        xx2 = right_trend[0, 0: int((seg_len + 1) / 2)]
        w = np.arange(0, nonoverlap_len + 1) / nonoverlap_len
        xx_right = np.multiply(xx1, (1 - w)) + np.multiply(xx2, w)

        record_x = np.hstack((record_x, xi_left[0, int((np.shape(xi_left)[1] + 3) / 2) - 1: np.shape(
            xi_left)[1]], xi_mid[0, int((np.shape(xi_mid)[1] + 1) / 2): np.shape(xi_mid)[1]]))
        record_y = np.hstack((record_y, xx_left[0, 1: np.shape(xx_left)[
                             1]], xx_right[0, 1: np.shape(xx_right)[1]]))

        right_start_index = np.mat([(j) for j in range(np.shape(
            xi_right)[1]) if xi_right[0, j] == xi_mid[0, np.shape(xi_mid)[1] - 1] + 1])
        nrows_mid = np.shape(right_start_index)[1]

        if nrows_mid == 1:
            right_start_index = right_start_index[0, 0]
            record_x = np.hstack(
                (record_x, xi_right[0, right_start_index: np.shape(xi_right)[1]]))
            record_y = np.hstack(
                (record_y, right_trend[0, right_start_index: np.shape(right_trend)[1]]))

        detrended_data = data - record_y.T

        return detrended_data, record_y


def multi_detrending(y, step_size, q, order):
    # y: input data,stored as a row or column vector
    # q: q spectrum

    q = np.mat(q)
    len = np.shape(y)[1]
    imax = int(round(np.log2(len)))
    #order = 2
    result = np.mat(
        np.zeros((np.shape(q)[1] + 1, int((imax - 2)/step_size) + 1)))
    k = 1
    for i in range(1, imax, step_size):
        w = int(round(2 ** i + 1))
        if w / 2 == 1:
            w = w + 1
        detrended_data, trend = detrending_method(y, w, order)
        result[0, k-1] = (w + 1)/2
        for j in range(1, np.shape(q)[1] + 1):
            # Euclidean norm
            abs_detrended_data = np.power(abs(detrended_data), q[0, j-1])
            Sum = abs_detrended_data.sum(
                axis=0) / (np.shape(detrended_data)[0] - 1)

            result[j, k-1] = Sum[0, 0] ** (1 / q[0, j - 1])

            # `result(j + 1,k) = (sum(abs(detrend_data - mean(detrended_data))) **q[j-1] /
            # ((shape(detrended_data)[0] - 1) ** (1/q[j-1]))`
            # earlier analysis suggests that without removing mean yield
            # more accurate estimate of H values

        k = k + 1

    result = np.log2(result)

    return result


def adaptive_filter(y, span=56):

    w = int(4 * np.floor(len(y)/span) + 1)
    y_dt = np.mat([float(j) for j in y])
    _, y_smooth = detrending_method(y_dt, w, 1)
    
    return y_smooth.T
