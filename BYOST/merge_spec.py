import numpy as np
from scipy import interpolate
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as itp
import matplotlib.pyplot as plt



## function to merge spectra
def merge_spec(wave_b,flux_b,wave_r,flux_r,interp_option=0,normalize='blue_side',plot_ax=None):
    """
    Input:
        wave_b,flux_b: Blue side spectrum
        wave_r,flux_r: Red side spectrum
        interp_option=0: wavelength grid interp, 0 as default combine the wavelength points in the overlapped region
        normalize: how flux is normalized when merging
                'blue_side'  match the flux of red spectrum to blue side in the overlapped region; 
                'red_side':  match the flux of blue spectrum  to the red spectrum in the overlapped region; 
        plot_ax: default None as no plot output, if given ax then will plot the spectra before and after the merging process. 
        
    Output:
        tunple: merged_wavelength, merged_flux 
    """
    ## copyed from Eric's IDL EYH_MERGE_SPEC
    weight_lo=0.
    w1 = np.where(wave_b>=min(wave_r))[0]
    w2 = np.where(wave_r<=max(wave_b))[0]
    if (len(w1)<2) or (len(w2)<2):
        print('WARNING! Not enough overlap')
        return
    else:
        ### decide what wavelegnth to use in the overlap region
        if interp_option ==0: ## combine the overlap wavelengths
            wave_overlap = np.concatenate([wave_b[w1], wave_r[w2]],axis=0)
            wave_overlap = np.sort(wave_overlap)
            wave_overlap = np.unique(wave_overlap)
        elif interp_option ==1:
            wave_overlap = wave_b[w1]
        elif interp_option ==2:
            wave_overlap = wave_r[w2]
        ### decide how to normalize
        if normalize == 'blue_side':
            ### match the overlappen region flux based on the blue side
            norm_b = 1
            norm_r = simps(flux_b[w1],wave_b[w1])/simps(flux_r[w2],wave_r[w2])
        elif normalize == 'red_side':
            norm_b = simps(flux_r[w2],wave_r[w2])/simps(flux_b[w1],wave_b[w1])
            norm_r = 1
        else:
            norm_b,norm_r = 1,1 ## let the spectra merge by weight, smooothly merging together
            
        ### inteplate the flux in overlaped region
        x = [min(wave_overlap), max(wave_overlap)]
        f1 = interpolate.interp1d(x, [1.,weight_lo])
        f2 = interpolate.interp1d(x, [weight_lo,1.])
        weight1 = f1(wave_overlap)
        weight2 = f2(wave_overlap)
        f1_f = interpolate.interp1d(wave_b[w1],norm_b*flux_b[w1],fill_value="extrapolate")
        f2_f = interpolate.interp1d(wave_r[w2],norm_r*flux_r[w2],fill_value="extrapolate")
        flux_ol = 1./(weight1+weight2)*(f1_f(wave_overlap)*weight1 + f2_f(wave_overlap)*weight2)
        
        ### now combine the flux 
        w1_rest = np.where(wave_b<min(wave_r))[0]
        w2_rest = np.where(wave_r>max(wave_b))[0]
        wave_out = np.concatenate([wave_b[w1_rest],wave_overlap,wave_r[w2_rest]],axis=0)
        flux_out = np.concatenate([norm_b*flux_b[w1_rest],flux_ol,norm_r*flux_r[w2_rest]],axis=0)
        
        if plot_ax is not None:
            plot_ax.plot(wave_b,norm_b*flux_b,'b',alpha=0.4)
            plot_ax.plot(wave_r,norm_r*flux_r,'r',alpha=0.4)
            plot_ax.plot(wave_out,flux_out,'k',alpha=0.2)
            
        return wave_out,flux_out