'''Tracking stage of the image analysis pipeline'''

import numpy as np

#Load dataset
from PyImages.metadata import Metadata

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

#Tracking
import pims
import trackpy as tp

'''
Finds the trajectories of every object from every frame in all the timelapse sets.
This method processes the whole set of timelapse images for one experimental setup
(For example, an experiment with )

@param: i - index for a given 3D stack of timelapse images in the whole experiment

@return:
Send filtered and unfiltered dataframes of tracked particles for each set of timelapse images to storage folder
'''

def track_intensities(i):

    #extract metadata from the three channels
    #Note: md = Metadata('path to images')

    well_arr_nuclear = md.stkread(Channel='DeepBlue', Position=positions[i])
    well_arr_viral = md.stkread(Channel='Cyan', Position=positions[i])
    well_arr_death = md.stkread(Channel='FarRed', Position=positions[i])

    #segement the whole well using nuclear channel
    wvt_nuclear = label_wvt(well_arr_nuclear)
    watershed_wvt_nuclear = watershed_wvt(well_arr_nuclear,wvt_nuclear)

    #extract viral/death features
    cell_features_viral = features(watershed_wvt_nuclear, well_arr_viral)
    cell_features_death = features(watershed_wvt_nuclear, well_arr_death)

    #dataframe with viral/death intensities
    cellFrames_viral = pd.concat(cell_frames(watershed_wvt_nuclear,well_arr_viral, cell_features_viral))
    cellFrames_death = pd.concat(cell_frames(watershed_wvt_nuclear,well_arr_death, cell_features_death))

    #correct for artifacts and false areas
    correctCellFrames_viral = cellFrames_viral.loc[(cellFrames_viral.Filled_Area>=100)]
    correctCellFrames_death = cellFrames_death.loc[(cellFrames_death.Filled_Area>=100)]

    #Generate and filter tracks of cells using viral features
    traj_viral = tp.link_df(correctCellFrames_viral, 50, memory=8)
    traj_viral = tp.filter_stubs(traj_viral, 5)

    #Generate and filter tracks of cells using death features
    traj_death = tp.link_df(correctCellFrames_death, 50, memory=8)
    traj_death = tp.filter_stubs(traj_death, 5)

    #save the trajectory dataframe
    traj_viral.to_pickle('path to image storage_'+ str(i))
    traj_death.to_pickle('path to image storage_'+ str(i))

    #filter to include only particles of trajectory => 35
    traj_viral_plus35 = particles_plus35(traj_viral)
    traj_death_plus35 = particles_plus35(traj_death)

    #get intensities for each cell at each frame
    traj_viral_intensities = intensities_df(traj_viral_plus35)
    traj_death_intensities = intensities_df(traj_death_plus35)

    #store plus35 intensities
    traj_viral_intensities.to_pickle('path to storage/Maxim/Segmented_Images/Viral_intensities_'+ str(i))
    traj_death_intensities.to_pickle('path to storage/Maxim/Segmented_Images/Death_intensities_'+ str(i))

'''
Find particles with a trajectory greater than 35 frames
Note: 35 frames is an arbitrary standard and can be modified accordingly

@param: df - pandas dataframe

@return: df_plus35 - a new dataframe with particles of trajectories less than 35 filtered out
'''
def particles_plus35(df):
    particle_counts = df.groupby(df.particle,as_index=False).size()
    df_plus35 = df.set_index(df.particle).loc[particle_counts.loc[particle_counts>35].index]
    return df_plus35

'''
Create a dataframe of intensities from trajectory dataframes

@param: df - trajectory dataframe filtered for trajectory length

@return: dataframe - intensities of particles for each frame in timelapse
'''
def intensities_df(df):
    #create new dataframe with particles as rows, and time frames as columns
    intensities = pd.DataFrame(index=df.particle.drop_duplicates().values, columns=df.frame.drop_duplicates().values)

    df = df.set_index(df.frame)
    #for loop that finds intensities for a particle through each frame
    for p in df.particle.drop_duplicates().values:
        #frame_intensities is a temporary dataframe that reduces df columns to only a particle label and 90th percentile intensity
        #Note:particle column not necessary and probably don't also need the new dataframe 'frame_intensities'
        frame_intensities = df.loc[df.particle==p,['particle','90_intensity']]

        #for each frame in frame_intensities, add the intensity of particle p into the intesities dataframe
        #if the particle has no intensities/disappeared at a given frame, it is null to keep dataframe size consistent.
        for f in df.frame.drop_duplicates().values:
            if f not in frame_intensities.index:
                intensities.loc[p,f] = None
            else:
                intensities.loc[p,f] = frame_intensities.loc[f, '90_intensity']

    #change the string intensities to numeric value for each col(frame) in intensities
    for col in intensities:
        intensities[col] = pd.to_numeric(intensities[col], errors='coerce')
    return intensities
