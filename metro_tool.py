# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import datetime
import copy

def MAE(target,predict):

    error = []
    for i in range(len(target)):
        error.append(target[i] - predict[i])

    absError = []
    for val in error:
        absError.append(abs(val))
    MAE = sum(absError) / len(absError)
    print('MAE:',MAE)
    return MAE

def abs_relative_error(y_pred, y_true):
    return np.mean(np.mean(np.abs(y_pred - y_true) / np.abs(y_pred + y_true)))

def abs_relative_error_element(y_pred, y_true):
    return np.abs(y_pred - y_true) / np.abs(y_pred + y_true)

def abs_error(y_pred, y_true):
    return np.mean(np.mean(np.abs(y_pred - y_true)))

def abs_error_element(y_pred, y_true):
    return np.abs(y_pred - y_true)
