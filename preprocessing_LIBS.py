"""
https://github.com/LIBS-ML-team/libs-transfer-library/blob/main/src/preprocessing.py
"""


import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from multiprocessing import Pool

def process_row(process, row):
    return process(row)

def apply_multiprocessing_along_axis(process, data):
    with Pool() as pool:
        result = pool.starmap(process_row, [(process, row) for row in data])

    return np.array(result)

class LabelCropp(BaseEstimator, TransformerMixin):
  """
  TODO 
  """
  def __init__(self, label_from=None, label_to=None, labels=None):
    self.label_from = label_from
    self.label_to = label_to
    self.labels = labels


  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self


  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    labels = self.labels if self.labels is not None else np.arange(self.n_features_in_)
    label_from = self.label_from if self.label_from is not None else labels[0]
    label_to = self.label_to if self.label_to is not None else labels[-1]
      
    if label_from > labels.max() or label_to < labels.min():
      warnings.warn('Labels out of range! Skipping!')
      return X

    idx_from, idx_to = np.argmax(labels >= label_from),  len(labels) - np.argmax((labels <= label_to)[::-1]) - 1
    return X[:, idx_from: idx_to]



def estimate_baseline(
    uc_spectrum,
    **kwargs
) -> np.ndarray :
  """
  Estimates the input spectrum's baseline by performing a sliding minimum
  operation followed by a sliding Gaussian mean smooth_minima

  Possible kwargs: ww_min, ww_smooth

  RETURN the estimated baseline <baseline>:np.array
  """

  if 'ww_min' not in kwargs: ww_min = 100
  else: ww_min = kwargs['ww_min']
  if 'ww_smooth' not in kwargs: ww_smooth = int(ww_min / 2)
  else: ww_smooth = kwargs['ww_smooth']
  if isinstance(uc_spectrum, np.ndarray):
      spectrum_proc = uc_spectrum.copy()
  else:
      spectrum_proc = uc_spectrum.to_numpy()
 
  assert len(spectrum_proc.shape) == 1, 'wrong input dimension'

  # spectrum_proc = spectrum_proc - np.min(spectrum_proc)
  
  baseline = smooth_minima(
      rolling_minima(
          spectrum_proc,
          window_width=ww_min
      ),
      window_width=ww_smooth
  )
  assert baseline.shape[0] == uc_spectrum.shape[0], \
    'Estimate Baseline processing error resulted in unexpected output shape'

  return(baseline)


def rolling_minima(
    spectrum_input,
    **kwargs
) -> np.ndarray :
  """
  RETURN the local minima of <spectrum_input> determined by a moving minimum
  filter of width <kwargs['window_width']>

  Possible kwargs: window_width
  """

  if 'window_width' not in kwargs: window_width = 200
  else: window_width = kwargs['window_width']

  if isinstance(spectrum_input, pd.core.series.Series):
      spectrum_proc = spectrum_input.copy()
  else:
      spectrum_proc = pd.Series(spectrum_input)  
  
  assert len(spectrum_proc.shape) == 1, 'wrong input dimension'

  minima_left = spectrum_proc.rolling(
      window=window_width,
      min_periods=1
  ).min()
  minima_right = spectrum_proc[::-1].rolling(
      window=window_width,
      min_periods=1
  ).min()[::-1]

  rolling_minima = np.array(
      [
          minima_left.to_numpy(), 
          minima_right.to_numpy()
      ]
  ).mean(axis=0)

  assert rolling_minima.shape[0] == spectrum_input.shape[0], \
    'Rolling Minima processing error resulted in unexpected output shape'

  return(rolling_minima)


def smooth_minima(
    spectrum_input,
    **kwargs
) -> np.ndarray :
  """
  RETURN <spectrum_input> smoothed by a sliding Gaussian weighted mean operator 

  Possible kwargs: window_width
  """

  if 'window_width' not in kwargs: window_width = 200
  else: window_width = kwargs['window_width']

  append_size = int(window_width)

  if isinstance(spectrum_input, pd.core.series.Series):
    spectrum_proc = spectrum_input.copy()
  else:
    spectrum_proc = pd.Series(spectrum_input)
  
  assert len(spectrum_proc.shape) == 1, \
    'wrong input dimension in baseline smooth_minima'

  spectrum_proc = pd.Series(
      np.zeros((append_size))
  )._append( #append -> _append
      spectrum_proc
  )._append( #append -> _append
      pd.Series(
        np.zeros((append_size))
      )
  )

  smooth_minima_left = spectrum_proc.rolling(
      window=window_width,
      min_periods=1,
      win_type='gaussian'
  ).mean(std=window_width)

  smooth_minima_right = spectrum_proc.iloc[::-1].rolling(
      window=window_width,
      min_periods=1,
      win_type='gaussian'
  ).mean(std=window_width)

  smooth_minima_right = smooth_minima_right.iloc[::-1]

  smooth_minima = smooth_minima_left[append_size:-append_size]
  smooth_minima += smooth_minima_right[append_size:-append_size]
  smooth_minima /= 2

  assert smooth_minima.shape[0] == spectrum_input.shape[0], \
    'smooth_minima processing error resulted in unexpected output shape'

  return(smooth_minima)


def correct_baseline(
  spectrum_input,
  **kwargs
) -> np.ndarray :
  """
  RETURN the baseline corrected spectrum

  Possible kwargs: ww_min, ww_smooth
  """

  if 'win_min' not in kwargs: kwargs['ww_min'] = 200  

  if 'ww_smooth' not in kwargs: kwargs['ww_smooth'] = kwargs['ww_min'] // 2

  if isinstance(spectrum_input, np.ndarray):
    spectrum_proc = spectrum_input.copy()
  else:
    spectrum_proc = spectrum_input.to_numpy()
  
  baseline_correcter_spectrum = spectrum_proc - estimate_baseline(
      spectrum_proc, 
      ww_min=kwargs['ww_min'],
      ww_smooth=kwargs['ww_smooth']
  )
  
  return(baseline_correcter_spectrum)


def match_wavelengths(left, right, left_calibration, right_calibration):
  """
  Resample two spectral datasets to have common calibration.
  [left1(left2,right1)left3] or [left1(left2,right1]right2)
  """
  if left_calibration[0] > right_calibration[0]:
    right, left, calibration = match_wavelengths(right, left, right_calibration, left_calibration)
    return left, right, calibration

  # find the calibration for each part
  # left from the intersection
  left1 = left_calibration[left_calibration < right_calibration[0]]
  # intersection
  left2 = left_calibration[(left_calibration >= right_calibration[0]) & (left_calibration <= right_calibration[-1])]
  # right from the intersection
  left3 = left_calibration[left_calibration > right_calibration[-1]]

  # intersection
  right1 = right_calibration[(right_calibration >= left_calibration[0]) & (right_calibration <= left_calibration[-1])]
  # right from the intersection
  right2 = right_calibration[right_calibration > left_calibration[-1]]

  # combine the calibrations using the finer one for the intersection
  if np.mean(np.diff(left2)) < np.mean(np.diff(right1)):
    calibration = np.hstack((left1, left2, right2, left3))
  else:
    calibration = np.hstack((left1, right1, right2, left3))

  # resample the spectra
  left = np.apply_along_axis(lambda x: np.interp(calibration, left_calibration, x, left=0., right=0.), 1, left)
  right = np.apply_along_axis(lambda x: np.interp(calibration, right_calibration, x, left=0., right=0.), 1, right)
  return left, right, calibration