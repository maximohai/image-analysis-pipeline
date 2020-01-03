'''Feature extraction stage of image analysis pipeline'''

import numpy as np

#Binary Masking
import pywt
from skimage.filters import threshold_local
from skimage.morphology import binary_opening,binary_closing, binary_erosion, disk

#Segmentation
from skimage.morphology import watershed, remove_small_objects
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi

#Feature extraction
from skimage.measure import regionprops
import pandas as pd
from pandas import DataFrame, Series

'''
extract features from labeled objects.

@param: watershed_wvt -
@param: well_arr -

@return: a list of properties for each object in each image in well.
'''

def features(watershed_wvt,well_arr):
    props = []
    for k in range(watershed_wvt.shape[2]):
        img_props = regionprops(watershed_wvt[:,:,k],intensity_image=well_arr[:,:,k])
        props.append(img_props)
    return props

'''
Create appropriately sized dataframe and fill it with features for each particle in timelapse

@param: watershed_wvt - nparray of well_arr image masks with labeled segmented particles
@param: well_arr - 3D stack of raw timelapse images
@param: cell_props - list of properties for each particle in each frame of watershed_wvt masks
                   // output from features() method

@return: Filled pd dataframe of properties/features for each particle in each image in timelapse stack
'''
def cell_frames(watershed_wvt,well_arr,cell_props):
    cellFrames = []
    nineties = ninety_percentile(well_arr, cell_props)
    #Create empty cellFrames dataframe
    for k in range(watershed_wvt.shape[2]):
        cell_frame = pd.DataFrame(index=range(1, watershed_wvt[:,:,k].max()+1), columns=['y','x','Filled_Area','90_intensity','frame'])
        cellFrames.append(cell_frame)
    #Fill the dataframe with properties for each particle
    for k in range(len(cellFrames)):
        cellFrames[k]['y'] = [prop.centroid[0] for prop in cell_props[k]]
        cellFrames[k]['x'] = [prop.centroid[1] for prop in cell_props[k]]
        cellFrames[k].Filled_Area = [prop.area for prop in cell_props[k]]
        cellFrames[k]['90_intensity'] = nineties[k]
        cellFrames[k]['frame'] = k

    return cellFrames

'''
Finds the 90th percentile intensities for every labeled particle in every frame of a timelapse.
This is in opposition to using the maximum intensity of a particle.

@param: well_arr - 3D stack of raw timelapse images
@param: cell_props - list of properties for each particle in each frame of watershed_wvt masks

@return: frames - a list of 90th percentile intensities for every particle in each frame of timelapse
'''
def ninety_percentile(well_arr, cell_props):
    frames = []
    for k in range(well_arr.shape[2]):
        percent90s = []
        for i in range(len(cell_props[k])):
            labeled_object = cell_props[k][i]
            percent90 = np.percentile(well_arr[np.where(labeled_object.image)][:,k], 90)
            percent90s.append(percent90)
        frames.append(percent90s)
    return frames
