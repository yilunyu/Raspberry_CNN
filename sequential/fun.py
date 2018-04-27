import numpy as np

#[b,h,w,d], where b is 1
def convert_tfinput_2_rasp(arr):
    s = len(arr.shape[1]*arr.shape[2]*arr.shape[3])
    output = np.zeros(s)
    for h,m1 in enumerate(arr[1]):
        for w,m2 in enumerate(m1):
            for d,value in enumerate(m2):
                index = h*arr.shape[1]*arr.shape[2]+w*arr.shape[2]+d
                output[index] = value
    return output


def convert_tfkernel_2_rasp(arr):
    s = len(arr.shape[0]*arr.shape[1]*arr.shape[2]*arr.shape[3])
    output = np.zeros(s)
    for h,m1 in enumerate(arr[0]):
        for w,m2 in enumerate(m1):
            for in_chan,m3 in enumerate(m2):
                for out_chan,value in enumerate(m3):
                    index = h*arr.shape[0]*arr.shape[1]*arr.shape[2]\
                        +w*arr.shape[1]*arr.shape[2]+in_chan*arr.shape[2] \
                        +out_chan
                    output[index] = value
    return output


