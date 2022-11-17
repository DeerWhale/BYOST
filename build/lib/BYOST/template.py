#!/usr/bin/env python

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import math
from astropy.io import fits as pyfits
from scipy import interpolate
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as itp
from scipy.ndimage import gaussian_filter1d

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import WhiteKernel,RBF,ConstantKernel

import os
base_path = os.path.dirname(__file__)

from .merge_spec import merge_spec



### function to get Hsiao template
def get_Hsiao_template(epoch):
    """
        Input:
            epoch: time since B-band maximum, interger within range -19 to 85
        
        Output:
            tuple: wave,flux
    
    """
    if int(epoch)<-19 or int(epoch)>85:
        raise ValueError('OH NO, epoch out of range of the Hsiao template (-19,85)!')
    
    ## read in the Hsiao templates
    f = pyfits.open('/'.join([base_path,'files/Hsiao_SED_V3_Ia.fits']))
    h_sed = f[0].data
    head = f[0].header
    h_wav = head['CRVAL1'] + (np.arange(head['NAXIS1'],dtype=np.float32) - head['CRPIX1'] + 1)*head['CDELT1']
    h_time = head['CRVAL2'] + (np.arange(head['NAXIS2'],dtype=np.float32) - head['CRPIX2'] + 1)*head['CDELT2']
    f.close()

    ## locate the ID of the epoch
    ID = list(h_time).index(int(epoch))
    flux = h_sed[ID]
    return h_wav,flux


def get_template(df_buildingblocks,condition1,condition2,
                 PC_select_method = ['GPR_score_threshold',0.2],
                 return_template_error=False,error_MC_num = 1000,return_MC_spectra=False):
    """
    Input:
        df_buildingblocks: pandas dataframe contains resulting PCA and GPR 
            ** if there is more than 1 wavelength region (df.shape[0]>1), the wavelength must be aranged from 
            blue to red in the dataframe, and has overlap in neribouring region in order to enable merging **
        condition1: input variable 1, scaler 
        condition2:  input variable 2, scaler 
        
        ** method of selecting which PCs to keep for template flux construction **
        PC_select_method = ['method',value]
            'GPR_score_threshold': keep PCs that has GPR R^2 >= GPR_score_threshold
            'PCA_variance_pctg_threshold': keep PCs up to the one that has total_variance >= PCA_variance_pctg_threshold
            'fixed_PC_number': keep n (n=fixed_PC_number) first PCs
        
        return_template_error: default False; If True, return the template flux error
        error_MC_num: will be used is return_template_error=True, the number of the 
                      interations to get the template flux error.
        return_MC_spectra: default False, if True, return all the possible spectra during the MC. 
    Output:
        tuple: template_wavelength, template_flux (if return_template_error=False)
        tuple: template_wavelength, template_flux, template_error (if return_template_error=True and return_MC_spectra=False)
        tuple: template_wavelength, template_flux, template_error, MC_template_flux (if return_template_error=True and return_MC_spectra=True) 
    """
    
    ## read the PC selecting method
    PC_selecting_methods = ['GPR_score_threshold','PCA_variance_pctg_threshold','fixed_PC_number']
    PC_selecting_methods_values = [None,None,None] ## correcsonding to above one by one
    PC_selecting_methods_values[PC_selecting_methods.index(PC_select_method[0])] = PC_select_method[1]
    GPR_score_threshold = PC_selecting_methods_values[0]
    PCA_variance_pctg_threshold = PC_selecting_methods_values[1]
    fixed_PC_number = PC_selecting_methods_values[2]
    
    ## chekc if there is more than one wavelength region
    if df_buildingblocks.shape[0]==1:
        return get_single_spectrum_template(df_buildingblocks,condition1,condition2,GPR_score_threshold=GPR_score_threshold,\
                                            PCA_variance_pctg_threshold=PCA_variance_pctg_threshold,fixed_PC_number=fixed_PC_number,\
                                            return_template_error = return_template_error,error_MC_num=error_MC_num,return_MC_spectra=return_MC_spectra)
    elif df_buildingblocks.shape[0]>1: ## merge the wavelength regions if there is more than 1 region
        ## first check if all the wavelength has overlap 
        df_buildingblocks = df_buildingblocks.reset_index(drop=True) #just in case the index is not continued
        for i, row in df_buildingblocks[:-1].iterrows():
            wave1,wave2 = row.wavelength, df_buildingblocks.wavelength[i+1]
            if wave1[-1] <= wave2[0]:
                raise ValueError(f'df_buildingblocks row index {i} and {i+1} do not have overlap! \
                Please fix that then we can stitch togehter the spectrum.')
        
        ## contruct template region by region and merge them 
        for i in range(df_buildingblocks.shape[0]):
            ## take the first one as the initial one
            if i==0: 
                template_merged = get_single_spectrum_template(df_buildingblocks[i:i+1],condition1,condition2,GPR_score_threshold=GPR_score_threshold,\
                                        PCA_variance_pctg_threshold=PCA_variance_pctg_threshold,fixed_PC_number=fixed_PC_number,\
                                        return_template_error = return_template_error,error_MC_num=error_MC_num,return_MC_spectra=return_MC_spectra)
                if return_template_error == True:
                    wave_SNR_merged,flux_SNR_merged = template_merged[0],template_merged[1]/template_merged[2]
                    wave_MC = template_merged[0]
            else:
                ## merge the rest onto the previous ones
                template = get_single_spectrum_template(df_buildingblocks[i:i+1],condition1,condition2,GPR_score_threshold=GPR_score_threshold,\
                                    PCA_variance_pctg_threshold=PCA_variance_pctg_threshold,fixed_PC_number=fixed_PC_number,\
                                    return_template_error = return_template_error,error_MC_num=error_MC_num,return_MC_spectra=return_MC_spectra)
                # merge flux 
                template_merged[0],template_merged[1] = merge_spec(template_merged[0],template_merged[1],template[0],template[1])
                # merge flux error if needed
                if return_template_error == True:
                    SNR = template[1]/template[2]
                    wave_SNR_merged,flux_SNR_merged = merge_spec(wave_SNR_merged,flux_SNR_merged,template[0],SNR,normalize = 0)
                    template_merged[2] = template_merged[1]/flux_SNR_merged
                    # merge the MC flux if needed
                    if return_MC_spectra==True:
                        for n in range(error_MC_num):
                            wave_temp,template_merged[3][n] = merge_spec(wave_MC,template_merged[3][n],template[0],template[3][n])
                        wave_MC = template_merged[0] 
        ## return the merge flux and errors            
        return template_merged
    else:
        raise ValueError('Please double check input df_buildingblocks (the first argument), it seems to be empty.')
    
    
##-----------------------------------------------------------------------------##

## function of predicting PCs given GPR_output from one set of pca
def GPR_predict_PC(GPR_output,condition1,condition2):
    """
    Input:
        GPR_output: a pandas dataframe of the fitted gp (and gp scores) for each PC
        condition_1: input variable 1, scaler 
        condition_2:  input variable 2, scaler
        
    Output:
        pred_PCs: predicted PC projections, list
        pred_PC_sigmas: predicted PC projection sigmas, list       
    """
    ## define empty lists for the output
    pred_PCs, pred_PC_sigmas = [],[]
    for PC in GPR_output.columns.values: 
        ## locate the original PC range and min to get the normalization factor for later
        yrange = GPR_output[PC]['yrange']
        ymin   = GPR_output[PC]['ymin']
        ## get the fitted gp
        gp = GPR_output[PC]['gp']
        if isinstance(condition1,(np.ndarray,list)) or isinstance(condition2,(np.ndarray,list)):
            raise ValueError('It seems like either condition1 or condition2 is list/array, but\
                              please change it to scaler, one value at a time please~')
        X = np.array([[condition1,condition2]])
        y_gp,y_sigma = gp.predict(X, return_std=True)
        ## unnormalize the predicted y using the y above
        y_gp_unnorm = y_gp*yrange+ymin
        y_sigma_unnorm = y_sigma*yrange
        pred_PCs.append(y_gp_unnorm.flatten()[0])
        pred_PC_sigmas.append(y_sigma_unnorm.flatten()[0])
        
    return pred_PCs,pred_PC_sigmas



## function of getting a single spectrum of templates from fitted pca and GPRs
def get_single_spectrum_template(df_buildingblock,condition1,condition2,
                                 GPR_score_threshold=0.2,PCA_variance_pctg_threshold=None,fixed_PC_number=None,\
                                 return_template_error = False,error_MC_num=1000,return_MC_spectra=False):  
    """
    Input:
        df_buildingblock: pandas dataframe contains resulting PCA and GPR 
        condition_1: input variable 1, scaler 
        condition_2:  input variable 2, scaler 
        
        ** method of selecting which PCs to keep for template flux construction **
        GPR_score_threshold: keep PCs that has GPR R^2 >= GPR_score_threshold
        PCA_variance_pctg_threshold: keep PCs up to the one that has total_variance >= PCA_variance_pctg_threshold
        fixed_PC_number: keep n (n=fixed_PC_number) first PCs
        
        return_template_error: default False; If True, return the template flux error
        error_MC_num: will be used is return_template_error=True, the number of the 
                      interations to get the template flux error.
        return_MC_spectra: default False, if True, return all the possible spectra during the MC. 
        
    Output:
        tuple: template_wavelength, template_flux (if return_template_error=False)
        tuple: template_wavelength, template_flux, template_error (if return_template_error=True and return_MC_spectra=False)
        tuple: template_wavelength, template_flux, template_error, MC_template_flux (if return_template_error=True and return_MC_spectra=True) 
    """
    df_buildingblock = df_buildingblock.reset_index(drop=True) # just in case
    ## check data type and of value is in range
    if isinstance(condition1,(list,np.ndarray)) or isinstance(condition2,(list,np.ndarray)):
        raise ValueError('Input single scaler for condition1 and condition2 please, one at a time :)')
    else:
        condition1_range = df_buildingblock.condition1_range[0]
        condition2_range = df_buildingblock.condition2_range[0]
        if (condition1 < condition1_range[0]) or (condition1 > condition1_range[1]):
            raise ValueError(f'Warning! condition1 not in modeled range, please input epoch between [{condition1_range[0]},{condition1_range[1]}]')
        if (condition2 < condition2_range[0]) or (condition2 > condition2_range[1]):
            raise ValueError(f'condition2 not in supported range, please input condition2 between [{condition2_range[0]},{condition2_range[1]}]')
    ## set up the input and output
    x1,x2 = condition1,condition2
    ## get pca and predicted PCs from GPR
    pca,template_wavelength = df_buildingblock.pca[0], df_buildingblock.wavelength[0].copy()
    GPR_output = df_buildingblock.GPR_output[0].copy()
    pred_PCs,pred_PC_sigmas = GPR_predict_PC(GPR_output,condition1,condition2)
    
    ## select which PCs to keep
    pred_PCs_allPC = pred_PCs.copy()
    if (int(GPR_score_threshold is not None))+int(PCA_variance_pctg_threshold is not None)+(fixed_PC_number is not None) >= 2:
            raise ValueError('Please choose EITHER GPR_score_threshold OR PCA_variance_threshold OR fixed_PC_number')
    if GPR_score_threshold is not None:
        gp_scores = GPR_output.loc['gp_score'].values
        GPR_score_threshold_bool = np.array([1. if (score >= GPR_score_threshold) else 0. for score in gp_scores]) 
        pred_PCs = pred_PCs_allPC*GPR_score_threshold_bool
    elif PCA_variance_pctg_threshold is not None:
        pca_total_pctgs = np.array([np.sum(pca.explained_variance_ratio_[:nn+1]*100) for nn in range(len(pca.explained_variance_ratio_))])
        keep_n_PC = len(pca_total_pctgs[pca_total_pctgs<PCA_variance_pctg_threshold])+1
        PCA_variance_pctg_threshold_bool = np.array([1. if (nn < keep_n_PC) else 0. for nn in range(pca.n_components_)]) 
        pred_PCs = pred_PCs_allPC*PCA_variance_pctg_threshold_bool
    elif fixed_PC_number is not None:
        fixed_PC_number_bool = np.array([1. if (nn < fixed_PC_number) else 0. for nn in range(pca.n_components_)]) 
        pred_PCs = pred_PCs_allPC*fixed_PC_number_bool
        
    ## construct template flux through inverse pca transformation     
    template_flux = pca.inverse_transform(pred_PCs)
    
    ## MC generate many spectrum within gp sigma and take std to the SNR
    if return_template_error is not False:
        flux_MC = []
        if (return_MC_spectra is True):
            MC_template_flux = [[] for i in range(error_MC_num)]
        for n in range(error_MC_num):
            pred_PCs_random = [np.random.normal(mean,pred_PC_sigmas[nnn]) for nnn,mean in enumerate(pred_PCs_allPC)]
            flux_MC.append(pca.inverse_transform(pred_PCs_random))
            if return_MC_spectra is True:
                MC_template_flux[n] = pca.inverse_transform(pred_PCs_random)
        flux_MC_std = np.array(flux_MC).std(axis=0)
        template_error = flux_MC_std
        
    ## return stuff
    if return_template_error is not False:
        if return_MC_spectra is True:
            return [template_wavelength,template_flux,template_error,MC_template_flux]
        else:
            return [template_wavelength,template_flux,template_error]
    else:
        return [template_wavelength,template_flux]

