from dezero.utils import pair, get_conv_outsize
from dezero import cuda

import numpy as np


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img,
                    ((0, 0),  (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                    mode = 'constant', constant_value=(0,))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lin = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        # transpose((0, 4, 5, 1, 2, 3)) : (N, C, KH, KW, OH, OW) -> (N, OH, OW, C, KH, KW)
        # reshape(N * OH *  OW, -1) : (N, OH, OW, C, KH, KW) -> (N*OH*OW, C*KH*KW)
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * OH * OW, -1)

    return col

def _im2col_gpu(img, kernel_size, stride, pad):
    """img2col function for GPU

    This code is ported form chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)

    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)

    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
            int c0 = i / (kh * kw * out_h * out_w);
            int ky = i / (kw * out_h * out_w) % kh;
            int kx = i / (out_h * out_w) % kw;
            int out_y = i / out_w % out_h;
            int out_x = i % out_w;
            int in_y = ky * dy + out_y * sy - ph;
            int in_x = kx * dx + out_x * sx - pw;
            if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w){
                col = img[in_x + w * (in_y + h * c0)]
            } else {
                col = 0;
            }
        ''',
        'im2col')(img.reduced_view(),
                 h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col
