''' Binary masking stage in the pipeline'''
import numpy as np

#Binary Masking
import pywt
from skimage.filters import threshold_local
from skimage.morphology import binary_opening,binary_closing, binary_erosion, disk

'''
Wavelet transform
Used to preprocess the image before binary masking

@param: img - 2D array image to be segmented
@param: keep_list - list of coefficents(?)
@param: wavelet - type of wavelet to be used in segmentation

@return: Result - 2D array of reconstructed image
'''
def wavelet_transform(img, keep_list=[3,4,5,6], wavelet='db9'):
    coeffs = pywt.wavedec2(img,wavelet)
    #returns list of coefficents
    for i in range(1,len(coeffs)+1):
        if i in keep_list:
            continue
        coeffs[-i] = tuple([np.zeros_like(v) for v in coeffs[-i]])
        #zero out the coefficients not in the keep list
    Result = pywt.waverec2(coeffs,wavelet)
    #reconstruct the 2D array of the image
    return Result

'''
Wavelet segmentation method - Courtesy of Evan Maltz
Creates the binary mask, using threshold_local and binary erosion

@param: img - image to be segmented
@param: keep - keep_list to be fed into wavelet_transform
@param: wv - wavelet to be fed into wavelet_transform
@param: disk_size - parameter for binary binary_erosion

@return: Binary mask (2D-array of bools)
'''
def wavelet_segment(img, keep=[3,4,5], wv='coif11', disk_size=6):
    #disk_size is a hyperparameter determined by fit to data
    img_wv = wavelet_transform(img, keep_list=keep, wavelet=wv)
    return binary_erosion(y>threshold_local(img_wv, block_size=311),disk(disk_size))
