'''Segmentation stage of the image analysis pipeline'''

import numpy as np
import random
import math

#Binary Masking
import pywt
from skimage.filters import threshold_local
from skimage.morphology import binary_opening,binary_closing, binary_erosion, disk

#Segmentation
from skimage.morphology import watershed, remove_small_objects
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi

'''
Label particles in stack of wavelet transformed binary masks

@param: well_arr - entire 3D time-lapse stack of images for one experimental set

@return: nparray of well_arr images with labeled particles, in correct 3D-orientation
'''
#generating labels
#returns list of labeled_wvt from image
def label_wvt(well_arr):
    labeled_wvts = []
    for i in range(well_arr.shape[2]):
        labeled_wvt = label(remove_small_objects(wavelet_segment(well_arr[:,:,i]), min_size=100))
        labeled_wvts.append(labeled_wvt)

    #transform list into 3D nparray in correct orientation
    labeled_wvts = np.stack(labeled_wvts)
    labeled_wvts = np.swapaxes(labeled_wvts.T, 0,1)
    return labeled_wvts

'''
Segment and label all particles in a stack of binary masks

@param: well_arr - 3D stack of unprocessed images
@param: wvt - 3D nparray of labeled binary masks with wavelet transform

@return: nparray of well_arr image masks with labeled segmented particles,
in correct 3D-orientation
'''
def watershed_wvt(well_arr,wvt):
    watershed_wvts = []
    for k in range(well_arr.shape[2]):
        img = gaussian_filter(well_arr[:,:,k],9)
        #gaussian_filter applid to raw image
        mask = wvt[:,:,k]
        _dte = ndi.distance_transform_edt(mask)
        _peaks = peak_local_max(img, indices=False, min_distance=5)
        #peak peak_local_max determined using gaussian blurred image
        watershed_img = watershed(-_dte,label(_peaks), mask=mask)
        watershed_wvts.append(watershed_img)

    #convert list into 3D nparray in correct orientation
    watershed_wvts = np.stack(watershed_wvts)
    watershed_wvts= np.swapaxes(watershed_wvts.T, 0,1)
    return watershed_wvts

'''
Recolor the labels for easier visualization

@param: labels - labeled mask

@return: recolor - binary mask with randomized label colors
'''
def recolor(labels):
    recolor = np.zeros(labels.shape)
    for i in range(1,labels.max()):
        rand = random.randint(100, 255)
        recolor[labels==i] = rand
    return recolor
