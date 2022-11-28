from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import math
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as itp
from scipy.ndimage import gaussian_filter1d

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import WhiteKernel,RBF,ConstantKernel



def make_buildingblock(df_spectra,df_conditions,\
                       normalize_method='intergrated_flux',normalize_wave_range=None, standardize_std = False,\
                       n_components=10,\
                       length_scales=[10.0,0.1],remove_outliars = True,\
                       n_restarts_optimizer=20):
    """
    Input:
        ** Input dataset and conditions wished to be modeld upon **
        df_spectra: pandas dataframe of the spectra on the common wavelenght grid, with wave as column names
        df_conditions: pandas dataframe of the conditions corresponding to df_spectra, e.g., epochs and sBVs
        
        ** arguements that could be used to prepare the data **
        normalize_method: default = 'intergrated_flux'; or None, "mean_flux" or "intergrated_flux"
            None: - None: take the input data as it is
            "mean_flux": normalize by dividing the mean flux in the selected range 
            "intergrated_flux": normalize by dividing the intergrated flux in the selected range
        normalize_wave_range: default = None; or 2-element list ([lambda_left,lambda_right])
        standardize_std: default = False, if=True, standardize the input data by standardeviation of each column
        
        ** arguement during the PCA step **
        n_components: default = 20, the number of the components you would like to keep for furhure analysis
        
        ** arguement during the GPR step**
        length_scales: default [10, 0.1], the length scale of the RBF kernal for condition_1 and condition_2
                       **The GPR depends on these initial scale values, try out the optiminal length scale 
                       for your data set!! (this is a little bit similar to the smoothness of the GP preditons, 
                       larger scale will return smoother precition, smaller scale will have more details)** 
        remove_outliars: default = True, ignore the local PC ourliars that are beyond 5sigma*global_std
        n_restarts_optimizer: number of restart of the optimizer
        
    Output:
        df_buildingblocks: pandas dataframe contains resulting PCA and GPR 
    """

    ## normalize the dataset if needed:
    if (normalize_method is not None):
        df_spectra = normalize_flux(df_spectra,normalize_method=normalize_method,normalize_wave_range=normalize_wave_range)

    ## Do PCA
    pca, PCA_projections, scaler = DO_PCA(df_spectra,n_components=n_components,standardize_std = standardize_std)

    ## Do GPR
    condition_1,condition_2 = df_conditions.T.values[0],df_conditions.T.values[1] # first and second column values
    GPR_output = DO_GPR(PCA_projections,condition_1,condition_2,length_scales=length_scales,\
                        remove_outliars = remove_outliars,\
                        n_restarts_optimizer=n_restarts_optimizer)

    ## store the results to a pandas dataframe, if would like to save the df, could save as a pickle file
    ## using df.to_pickle('filename.pkl'), which worked for me since it can keep the complex data structures
    df_buildingblock = pd.DataFrame({'wavelength':[df_spectra.columns.values],'scaler':[scaler],\
                                     'pca':[pca],'PCA_projections':[PCA_projections],'GPR_output':[GPR_output],\
                                     'condition1_range':[[min(df_conditions.T.values[0]),max(df_conditions.T.values[0])]],\
                                     'condition2_range':[[min(df_conditions.T.values[1]),max(df_conditions.T.values[1])]]},\
                                     index=[0])
    
    return df_buildingblock


## ------------------------------------------------------------------------------------##


## Normalize input spectrum to its mean or intergrated flux in certain wavelength range
def normalize_flux(df_spectra,normalize_method='mean_flux',normalize_wave_range=None):
    """
    Input:
        df_spectra: pandas dataframe of the spectra on the common wavelenght grid, with wave as column names
        normalize_method: "mean_flux" or "intergrated_flux"
            "mean_flux": normalize by dividing the mean flux in the selected range 
            "intergrated_flux": normalize by dividing the intergrated flux in the selected range
        normalize_wave_range: None or 2-element list ([lambda_left,lambda_right])
    Output:
        df_spectra: same format as input df_spectra but now each spectrum are normalized 
    """
    wave = df_spectra.columns.values
    for i,row in df_spectra.iterrows():
        ## select wave range
        if normalize_wave_range is None:
            lambda_left,lambda_right = wave[0],wave[-1]
        else:
            if len(normalize_wave_range) == 2:
                lambda_left,lambda_right = normalize_wave_range[0],normalize_wave_range[1]
            else:
                raise ValueError('normalize_wave_range should be 2-element [lambda_left,lambda_right] if not None')
        w = np.where((wave>=lambda_left)&(wave<=lambda_right))
        flux = row.values
        ## compute normalization factor
        if normalize_method == 'mean_flux':
            norm_factor = np.mean(flux[w])
        elif normalize_method == 'intergrated_flux':
            norm_factor = simps(flux[w],wave[w])
        else:
            norm_factor = 1
        ## normalize each spectrum
        df_spectra.loc[i,:] = df_spectra.loc[i,:]/norm_factor
        
    return df_spectra
        


## Perform PCA on the input data
def DO_PCA(PCA_input,n_components=10,standardize_std = False):
    """
    Input:
        PCA_input: pandas dataframe/2-D arrays of the normalized flux 
        n_components: default = 20, the number of the components you would like to keep for furhure analysis
        standardize_std: default = False, if=True, standardize the input data by standardeviation of each column
    Output:
        pca: fitted pca, see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        PCA_output: pandas dataframe of the PC projections, which has the same dimension as PCA_input in row, 
                    but dimension in columns is reduced to n_components)
    """
    #### standardize data if needed
    ## pca in sklearn take out the mean anyway so no need to standardize by mean in each column
    scaler = StandardScaler(with_mean=False, with_std=standardize_std)
    PCA_input_normed = scaler.fit_transform(PCA_input)
    ## perform PCA
    pca = PCA(n_components=n_components)  
    PCA_output = pca.fit_transform(PCA_input_normed)
    ## put results into a dataframe for convience (love pandas!)
    PC_names = ['PC'+str(i+1) for i in range(pca.n_components_)]
    PCA_projections = pd.DataFrame(PCA_output,columns=PC_names)
    
    return pca,PCA_projections,scaler


## Function of Gaussian process regresssion with a 2D input
def GPR_2D_input(x1,x2,y,yerr=None,length_scales=[10.0,0.1],n_restarts_optimizer=20,return_score=True):
    """
    Input:
        x1: input variable 1, N-elements 1-D array
        x2: input variable 2, N-elements 1-D array
        y:  dependent variable, N-elements 1-D array
        yerr: If not None, the errors of the dependent variable, N-elements 1-D array
        length_scales: default [10, 0.1], the length scale of the RBF kernal for x1 and x2
        n_restarts_optimizer: number of restart of the optimizer
        return_score: default True, return the GPR R^2 score on the predictions of y given x1 and x2
    Output:
        gp: fitted gp, see https://scikit-learn.org/stable/
            modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        gp_score (if return_score=True): scaler, the GPR R^2 score on the predictions of y given x1 and x2
                                         should be between 0 to 1, close to 1 is better generally. 
    """
    ### Do GPR on 2D input (epoch nad sBV(temp)) to predict PC
    X = np.array([x1, x2]).T
    ## use the fit_PC values as dependent y
    Y = y.reshape(-1,1) 
    if yerr is None:
        Yerr = 0.2*np.array([y[(x1>=x1[i]-10)&(x1<=x1[i]+10)].std() for i in range(len(y))])
    else:
        Yerr = yerr
    ## build the GPR kernal, RBF allows different length scale for individual input dimension, see post below
    # https://stackoverflow.com/questions/54604105/training-hyperparameters-for-multidimensional-gaussian-process-regression
    # Note that the GPR resutls is hightly dependent on the kernal design
    kernel = ConstantKernel(1.0, (1e-18, 1e18)) * RBF(length_scale = length_scales, length_scale_bounds=(1e-3 , 1e3)) + WhiteKernel(noise_level=np.std(y),noise_level_bounds=(1e-18 , 1e18))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer,alpha=Yerr**2,normalize_y=False)
    gp.fit(X,Y)
    if return_score == True:
        return gp,gp.score(X,Y)
    else:
        return gp 



def DO_GPR(PCA_projections,condition_1,condition_2,length_scales=[10.0,0.1],remove_outliars = True,\
           n_restarts_optimizer=20):
    """
    Input:
        PCA_output: pandas dataframe of the PCA projections
        condition_1: input variable 1, N-elements 1-D array
        condition_2:  input variable 2, N-elements 1-D array
        length_scales: default [10, 0.1], the length scale of the RBF kernal for condition_1 and condition_2
                       **The GPR depends on these initial scale values, try out the optiminal length scale 
                       for your data set!! (this is a little bit similar to the smoothness of the GP preditons, 
                       larger scale will return smoother precition, smaller scale will have more details)** 
        remove_outliars: default = True, ignore the local PC ourliars that are beyond 5sigma*global_std
        n_restarts_optimizer: number of restart of the optimizer
    Output:
        GPR_output: a pandas dataframe of the fitted gps (and gp scores if True) for each PC column 
                    given the conditions as inut
    """    
    ## define a empty df to store the results later
    GPR_output = pd.DataFrame()
    ## fitting GPR on every PC
    columns = PCA_projections.columns.values
    for i in tqdm(range(len(columns)), desc="Gaussian Process"):
        col = columns[i]
        y = PCA_projections[col].values
        x1,x2 = np.array(condition_1),np.array(condition_2)
        if remove_outliars==True:
            ## ignore the PC ourliars that are beyond 5 sigma of the local smoothed value
            smooth_y = gaussian_filter1d(y,sigma=5)
            y_diff = np.abs(y-smooth_y)
            w = np.where((y_diff<=5*np.std(y-smooth_y)))
            x1,x2,y = x1[w],x2[w],y[w] 
        ## Normalize the y to be in between 0 and 1  (it helps the gp_score to stay reseasonable)
        y_norm = (y-np.min(y))/(np.max(y)-np.min(y))
        ## GPR fit   
        gp_output = GPR_2D_input(x1,x2,y_norm,length_scales=length_scales,n_restarts_optimizer=n_restarts_optimizer,\
                          return_score=True)
        ## store the results
        GPR_output.loc['gp',col] = gp_output[0]
        GPR_output.loc['gp_score',col] = gp_output[1]
        GPR_output.loc['yrange',col] = np.max(y)-np.min(y)
        GPR_output.loc['ymin',col] = np.min(y)
    
    return GPR_output


