from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import math
from pylab import *
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import seaborn as sns
sns.set(style="ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

from .template import get_template,get_Hsiao_template


## plot the 2d histogram of the given conditions
def hist_2d(df_conditions,bins_condition1=None,bins_condition2=None):
    """
    Input:
        df_conditions: pandas dataframe of the conditions corresponding to df_spectra, e.g., epochs and sBVs
        bins_condition1: the bins of the hist for condition 1 
        bins_condition2: the bins of the hist for condition 2 
    
    Output:
        fig of the 2D histogram of the given conditions
    """    
    
    ## read in the columns 
    condition1 = df_conditions.T.values[0]
    condition2 = df_conditions.T.values[1]
    column_names = df_conditions.columns.to_list()
    
    ## decide bins for each of the hist
    if bins_condition1 is None:
        bins_condition1 = np.linspace(min(condition1),max(condition1),30)
    if bins_condition2 is None:
        bins_condition2 = np.linspace(min(condition2),max(condition2),30)

    #### plot the 2D hist
    f = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(6, 6)

    ax_joint = f.add_subplot(gs[1:, :-1])
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
    ax_marg_x.tick_params(labelbottom=False)#,labelleft=False,labelright=True)
    ax_marg_y.tick_params(labelleft=False,labelbottom=False, labeltop=True)

    joint_counts = ax_joint.hist2d(df_conditions[column_names[0]],df_conditions[column_names[1]],\
                                   bins = (bins_condition1,bins_condition2),cmap=plt.cm.Blues,alpha=0.6,zorder=0)
    ## mark the number of spectra 
    for i,x in enumerate(bins_condition1[:-1]):
        for j,y in enumerate(bins_condition2[:-1]):
            if joint_counts[0][i][j]>0:
                spec_count_text_1 = x+(bins_condition1[i+1]-bins_condition1[i])/2
                spec_count_text_2 = y+(bins_condition2[i+1]-bins_condition2[i])/2
                ax_joint.text(spec_count_text_1,spec_count_text_2,'%d'%(int(joint_counts[0][i][j])),fontsize=15,ha='center',va='center')
    ax_marg_x.hist(df_conditions[column_names[0]],bins = bins_condition1,color = 'tab:blue',alpha=0.5,)
    ax_marg_y.hist(df_conditions[column_names[1]],bins = bins_condition2,color = 'tab:blue',alpha=0.5,orientation="horizontal")
    ax_joint.set_xlabel(column_names[0],fontsize=25)
    ax_joint.set_ylabel(column_names[1],fontsize=25)
    for ax in [ax_joint,ax_marg_x,ax_marg_y]:
        ax.tick_params(labelsize=18)
    ax_joint.xaxis.set_ticks_position('both')
    ax_joint.yaxis.set_ticks_position('both')
    ax_marg_x.yaxis.set_ticks_position('both')
    ax_marg_y.xaxis.set_ticks_position('both')
    ax_marg_y.tick_params(labelsize=18,pad=5)
    
    
    
### plot the PCA results (PC vectors and projections, and the explained ratios)
def plot_PCA(df_buildingblock,df_conditions,n_components=None,PC_vector_sigma=1):
    """
    Input:
        df_buildingblock: pandas dataframe contains resulting PCA and GPR
        df_conditions: pandas dataframe of the conditions corresponding to df_spectra, e.g., epochs and sBVs
        n_components: number of PCs to plot, default all of PCs
        PC_vector_sigma: default 1, the PC projection sigma when plotting the variance represented by the PC vectors
    
    Output:
        Fig of PCA results (first row is PC vectors and its variation, \
                            second row is PC projections vs given conditions)
    """
    
    df_buildingblock = df_buildingblock.reset_index(drop=None) ### just in case
    
    ## how many PCs to plot
    if n_components is None:
        n_components = df_buildingblock.pca[0].n_components_
        
    ## creat the figure
    fig = plt.figure(figsize=(6*n_components,14*df_buildingblock.shape[0]))
    plt.subplots_adjust(hspace=0.8,wspace=0.1)
    gs = plt.GridSpec(7*df_buildingblock.shape[0],10*n_components+2)
    axes_PC_vec  = [[fig.add_subplot(gs[7*i:7*i+3,  10*j:10*j+10]) for j in range(n_components)] for i in range(df_buildingblock.shape[0])]
    axes_PC_proj = [[fig.add_subplot(gs[7*i+3:7*i+6,10*j:10*j+10]) for j in range(n_components)] for i in range(df_buildingblock.shape[0])]
    
    ## plot region by region
    for i in tqdm(range(df_buildingblock.shape[0]),desc='Wavelength regions'):
        wave = df_buildingblock.wavelength[i]
        scaler = df_buildingblock.scaler[i]
        pca = df_buildingblock.pca[i]
        mean_vector, pc_pctgs = pca.mean_,pca.explained_variance_ratio_*100
        pca_proj = df_buildingblock.PCA_projections[i]
        condition2_color_bins = np.linspace(np.min(df_conditions.T.values[1]),np.max(df_conditions.T.values[1]),100)
        condition2_color = sns.color_palette('viridis',len(condition2_color_bins))
        for j in tqdm(range(n_components),desc='PCs'):
            PC = 'PC'+str(j+1)
            ## plot the eigen vector
            ax = axes_PC_vec[i][j]
            PC_eigen_vec = pca.components_[j]
            norm = 0.3/np.max(np.abs(PC_eigen_vec))
            ax.plot(wave,PC_eigen_vec*norm,color='k',alpha=0.8,label='eigen vector',zorder=4)
            ax.plot(wave,[0 for x in wave],color='grey',alpha=0.7,zorder=1,lw=1.5,ls=':')   
            ## plot mean_verctor + eigenvector*random_cofficient
            variance = PC_vector_sigma * pca_proj[PC].std()
            PC_proj_random = np.linspace(-variance,+variance,100)
            cl_ptn = sns.color_palette('rainbow',len(PC_proj_random))
            norm_factor = np.max([scaler.inverse_transform([mean_vector+variance*PC_eigen_vec])[0],\
                                  scaler.inverse_transform([mean_vector-variance*PC_eigen_vec])[0]])
            ax.plot(wave,scaler.inverse_transform([mean_vector])[0]/norm_factor,color='k',ls='--',alpha=0.8,\
                    zorder=2,lw=2,label='mean vector')
            for k,coff in enumerate(PC_proj_random):
                total_vector = scaler.inverse_transform([mean_vector + coff*PC_eigen_vec])[0]/norm_factor
                ax.plot(wave,total_vector,color=cl_ptn[k],alpha=0.7,zorder=1,lw=1)    
            ## label the PC explained variance percentage    
            if j==0:
                ratio = '%s=%.2f%%'%(PC,pc_pctgs[j])
                ax.set_ylabel('Normalized PC vectors',fontsize=20)
                ax.legend(fontsize=18,loc='lower right',framealpha=0.5)
            else:
                ratio = '%s=%.2f%% ($\Sigma_{1}^{%d}PC_i$=%.2f%%)'%(PC,pc_pctgs[j],j+1,np.sum(pc_pctgs[0:j+1]))
            ax.set_title(ratio,fontsize=20,color='k',va='center')
            ax.set_ylim(-0.4,1.09)
            ax.set_xlim(wave[0],wave[-1])
            ax.set_xlabel('Wavelength',fontsize=20)
            
            ## plot projection vs condition1 and color-code by condition2
            ax = axes_PC_proj[i][j]
            PC_proj = pca_proj[PC].values/pca_proj[PC].std()
            proj_colors = [condition2_color[np.argmin(np.abs(condition2_color_bins-x))] for x in df_conditions.T.values[1]]
            ax.scatter(df_conditions.T.values[0],PC_proj,c=proj_colors,alpha=0.7)
            ax.set_xlabel(df_conditions.columns[0],fontsize=20)
            if j==0:
                ax.set_ylabel('PC projections/$\sigma$',fontsize=20)

        ## add color bars 
        # the PC proj variance color bar for the PC vector
        cax = fig.add_subplot(gs[7*i:7*i+3, -1])
        cax_values = np.linspace(-PC_vector_sigma,+PC_vector_sigma,100)
        normalize = mcolors.Normalize(vmin=np.min(cax_values), vmax=np.max(cax_values))
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.rainbow)
        scalarmappaple.set_array(cax_values)
        cb = fig.colorbar(scalarmappaple,cax=cax,orientation='vertical')
        cax.tick_params(labelsize=20)
        cb.set_label(label=r'x: $\vec{v}_{mean}+x\,\sigma\,\vec{v}_{eigen}$',fontsize = 25)
        # the PC proj variance color bar for the PC vector
        cax = fig.add_subplot(gs[7*i+3:7*i+6, -1])
        cax_values = condition2_color_bins
        normalize = mcolors.Normalize(vmin=np.min(cax_values), vmax=np.max(cax_values))
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.viridis)
        scalarmappaple.set_array(cax_values)
        cb = fig.colorbar(scalarmappaple,cax=cax,orientation='vertical')
        cax.tick_params(labelsize=20)
        cb.set_label(label=df_conditions.columns[1],fontsize = 25)

                
    ## formart axis            
    for ax in np.concatenate([axes_PC_vec,axes_PC_proj]).flatten():
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=16,labelleft=False)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
    for i in range(df_buildingblock.shape[0]):   
        ## plot yticks for the first subplot in the row
        axes_PC_vec[i][0].tick_params(labelleft=True)
        axes_PC_proj[i][0].tick_params(labelleft=True)
        ## adjust y lim for PC projections
        ymins,ymaxs = [],[]
        for j in range(n_components): 
            ymin,ymax = axes_PC_proj[i][j].get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
        for j in range(n_components): 
            ax = axes_PC_proj[i][j]
            ax.set_ylim(min(ymins),max(ymaxs))
            xmin,xmax = ax.get_xlim()
            ax.set_xlim(xmin-(xmax-xmin)/50,xmax+(xmax-xmin)/50)

    ## plot some distinguishing background colors if there are more than 1 wavelength region
    if df_buildingblock.shape[0]>1:
        Wblock_cl =sns.color_palette("muted",df_buildingblock.shape[0]) 
        for i in range(df_buildingblock.shape[0]):
            for ax in np.concatenate([axes_PC_vec[i],axes_PC_proj[i]]):
                ymin,ymax = ax.get_ylim()
                xmin,xmax = ax.get_xlim()
                wave = df_buildingblock.wavelength[i]
                ax.fill_betweenx([ymin,ymax],xmin,xmax,alpha=0.06,color=Wblock_cl[i])
                
    return fig



### plot the GPR results
def plot_GPR(df_buildingblock,df_conditions,Wave_bin_ID=0,PC='PC1',view_angle = [32,300]):
    """
    Input:
        df_buildingblock: pandas dataframe contains resulting PCA and GPR
        df_conditions: pandas dataframe of the conditions corresponding to df_spectra, e.g., epochs and sBVs
        Wave_bin_ID: default 0, the row index of the df_buildingblock
        PC: default 'PC1'
        view_angle: default [32,300],the viewing angle of the 3D plot 
    
    Output:
        Fig of GPR results, a 3D illustration of the GP fits and the 2D projections on the back
    """
    ## read the info need for GP
    df_bin = df_buildingblock.loc[Wave_bin_ID]
    gp = df_bin.GPR_output.loc['gp',PC]
    gp_score = df_bin.GPR_output.loc['gp_score',PC]
    yrange = df_bin.GPR_output.loc['yrange',PC]
    ymin = df_bin.GPR_output.loc['ymin',PC]
    
    ## read the input data for GP
    x1, x2 = df_conditions.T.values[0],df_conditions.T.values[1]
    y = df_bin.PCA_projections[PC].values
    
    ## prepare the prediction data for plotting
    size = 50
    new_x1,new_x2 = np.linspace(min(x1),max(x1),size),np.linspace(min(x2),max(x2),size)
    plot_x1, plot_x2 = np.meshgrid(new_x1,new_x2)
    gp_Y_pred, Y_sigma = gp.predict(np.array([plot_x1.flatten(), plot_x2.flatten()]).T, return_std=True)
    plot_Y = np.array([gp_Y_pred.flatten()[size*i:size*(i+1)] for i in range(size)])
    plot_Y = plot_Y*yrange+ymin ## unnorm the predictions
    
    #### define the 3D plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    
    ## plot the data and predicted surface
    ax.plot_surface(plot_x1, plot_x2, plot_Y,alpha=0.2,rstride=4, cstride=4,color='tab:blue')
    colors_values= (x2-min(x2))/(max(x2)-min(x2))
    ax.scatter(x1,x2,y,c=colors_values,cmap='viridis',marker='o',alpha=0.7) ## plot data points
    ax.set_xlabel(df_conditions.columns[0],fontsize=20,labelpad=20)
    ax.set_ylabel(df_conditions.columns[1],fontsize=20,labelpad=20)
    ax.set_zlabel(PC+' projections',fontsize=20,labelpad=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3)
    projection_2d_offset = max(x2)+(max(x2)-min(x2))/2
    ax.contour(plot_x1, plot_x2, plot_Y, zdir='y', offset=projection_2d_offset, cmap=cm.viridis,alpha=0.8)
    ax.scatter(x1,[projection_2d_offset for item in x1],y,c=colors_values,cmap='viridis',marker='.',alpha=0.4)
    normalize = mcolors.Normalize(vmin=np.min(x2), vmax=np.max(x2))
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap='viridis')
    scalarmappaple.set_array(x2)
    ax.set_title(r'Wave bin%s, %s ($R^2=%.2f$)'%(Wave_bin_ID,PC,gp_score),fontsize=20,y=0.98)
    cb = fig.colorbar(scalarmappaple, shrink=0.3, aspect=15,label=df_conditions.columns[1],pad=-1.05)
    cb.ax.tick_params(labelsize=16) 
    cb.ax.set_ylabel(df_conditions.columns[1], fontsize=22)
    ax.view_init(view_angle[0], view_angle[1])
    plt.draw()
    
    return fig



### plot the GPR score
def plot_GPR_score(df_buildingblock):
    """
    Input:
        df_buildingblock: pandas dataframe contains resulting PCA and GPR
    
    Output:
        Fig of GPR scores R^2 (range from 0 to 1, closer to 1 is better)
    """
    
    ## get the gp_scores
    n_components = df_buildingblock.pca[0].n_components_
    W_bins = ['W'+str(i) for i in range(df_buildingblock.shape[0])]
    PCs = ['PC'+str(i+1) for i in range(n_components)]
    gp_scores = [[] for i in W_bins]
    for i,W_bin in enumerate(W_bins):
        for j,PC in enumerate(PCs):
            gp_score = df_buildingblock.loc[i].GPR_output.loc['gp_score',PC]
            gp_scores[i].append(gp_score)

    df_gp_scores = pd.DataFrame({Wbin:gp_scores[i] for i,Wbin in enumerate(W_bins)},index=PCs)
    df_gp_scores = df_gp_scores.round(2).replace(0.00,0).replace(-0.00,0)
        
    ## creat the figure
    fig = plt.figure(figsize=(0.8*df_buildingblock.shape[0],0.7*n_components))
    plt.subplots_adjust(hspace=0.0,wspace=0.2)
    gs = plt.GridSpec(1,df_buildingblock.shape[0]*2+1)
    ax=fig.add_subplot(gs[:-1]) 
    cax = fig.add_subplot(gs[-1])
    
    ## plot the heatmap use seaborn
    sns.heatmap(df_gp_scores,annot=True, ax=ax,cbar_ax=cax,cmap='BuPu',annot_kws={'fontsize':13},\
            cbar_kws={'label':'GPR score ($R^2$)'},linewidths=.5)
    ax.tick_params(labelsize=15,labeltop=True,direction='in')
    ax.tick_params("x",pad=20)
    cax.tick_params(labelsize=15)
    cax.yaxis.label.set_size(18)
    ax.set_xticklabels(df_gp_scores.columns, va='center',minor=False)                      
                          
    return fig


### plot the template it self
def plot_template(df_buildingblock,df_conditions,
                  matching_wave_position=None,y_offset_gap = 1.0,
                  condition1_sample=None,condition2_sample=None,\
                  condition1_colormap = 'rainbow',condition2_colormap = 'viridis',
                  log_x=True, log_y=True):
    """
    Input:
        df_buildingblock: pandas dataframe contains resulting PCA and GPR
        df_conditions: pandas dataframe of the conditions corresponding to df_spectra, e.g., epochs and sBVs
        matching_wave_position: default will match the flux at the median wavelength postion
        y_offset_gap: the offset in yaxis between the sampling template
        condition1_sample: default will sample the mean-std, mean, mean+std of the condition1 values while varying condition2
        condition2_sample: default will sample the mean-std, mean, mean+std of the condition2 values while varying condition1
        condition1_colormap: the color secheme for varying condition1, default rainbow
        condition2_colormap: the color secheme for varying condition1, default viridis
        log_x: default True, wavelength is plotted in log scale
        log_y: default True, flux is plotted in log scale
        
    Output:
        Fig of template variation, with 2 panels:
            left panel:  variation within condition1 while keeping condition2 fixed at certain values
            right panel: variation within condition2 while keeping condition1 fixed at certain values    
    """
    ## read in the condition values and set up corresponding colors
    condition1,condition2 = df_conditions.T.values[0],df_conditions.T.values[1]
    condition1_bins = np.linspace(np.min(condition1),np.max(condition1),50)
    condition2_bins = np.linspace(np.min(condition2),np.max(condition2),50)
    cl_condition1 = sns.color_palette(condition1_colormap,len(condition1_bins))    
    cl_condition2 = sns.color_palette(condition2_colormap,len(condition2_bins))
    
    ## define the flux matching wave position if not given
    wave,flux = get_template(df_buildingblock,condition1[0],condition2[0],return_template_error=False)
    if matching_wave_position is None:
        matching_wave_position = np.median(wave)
    text_label_wave = np.median(wave)
    
    ## define the sampleing point if not given
    if condition1_sample is None:
        condition1_sample = [np.mean(condition1)-np.std(condition1),np.mean(condition1),np.mean(condition1)+np.std(condition1)]
    if condition2_sample is None:
        condition2_sample = [np.mean(condition2)-np.std(condition2),np.mean(condition2),np.mean(condition2)+np.std(condition2)]
    
    ## define the figure and axes
    fig,axes = plt.subplots(2,46,figsize=(22,12))
    plt.subplots_adjust(hspace=0.0,wspace=0.2)
    gs = axes[0][0].get_gridspec()
    for ax in axes.flatten(): ax.remove()
    ax1 = fig.add_subplot(gs[:,0:19])
    ax2 = fig.add_subplot(gs[:, 26:45])
    cax1 = fig.add_subplot(gs[:,19])
    cax2 = fig.add_subplot(gs[:,45])

    ## plot the colorbar
    for cax,values,colormap,label in zip([cax1,cax2],[condition1_bins,condition2_bins],[cm.get_cmap(condition1_colormap),cm.get_cmap(condition2_colormap)],df_conditions.columns.to_list()):
        normalize = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(values)
        cb = fig.colorbar(scalarmappaple,cax=cax,orientation='vertical')
        cax.tick_params(labelsize=20)
        cb.set_label(label=label,fontsize = 26)
        
    ## plot template variation with condition1 at certain fixed condtion2 
    for j in tqdm(range(len(condition2_sample)),desc='sampling condition1'):
        cond2 = condition2_sample[j]
        offset = y_offset_gap*(len(condition2_sample)-j)
        ## plot the template
        for i,cond1 in enumerate(condition1_bins):
            wave,flux = get_template(df_buildingblock,cond1,cond2,return_template_error=False)
            if log_y==True:
                plot_flux = np.log10(flux/np.mean(flux))+offset
            else:
                plot_flux = flux/np.mean(flux)+offset
            if i == 0:
                flux_match = plot_flux[np.argmin(np.abs(wave-matching_wave_position))]
            else:
                plot_flux = plot_flux - plot_flux[np.argmin(np.abs(wave-matching_wave_position))] + flux_match
            ax1.plot(wave,plot_flux,color = cl_condition1[i],alpha=0.3,lw=1.)
        ## label the showcase condtion2
        text = r'%s=%.2f'%(df_conditions.columns[1],cond2)
        ax1.text(text_label_wave,flux_match+0.2,text,color='k',alpha=0.8,fontsize=24,ha='left',va='bottom')
            
    ## plot template variation with condition2 at certain fixed condtion1
    for j in tqdm(range(len(condition1_sample)),desc='sampling condition2'):
        cond1 = condition1_sample[j]
        offset = y_offset_gap*(len(condition1_sample)-j)
        ## plot the template
        for i,cond2 in enumerate(condition2_bins):
            wave,flux = get_template(df_buildingblock,cond1,cond2,return_template_error=False)
            if log_y==True:
                plot_flux = np.log10(flux/np.mean(flux))+offset
            else:
                plot_flux = flux/np.mean(flux)+offset
            if i == 0:
                flux_match = plot_flux[np.argmin(np.abs(wave-matching_wave_position))]
            else:
                plot_flux = plot_flux - plot_flux[np.argmin(np.abs(wave-matching_wave_position))] + flux_match
            ax2.plot(wave,plot_flux,color = cl_condition2[i],alpha=0.3,lw=1.)
        ## label the showcase condtion1
        text = r'%s=%.2f'%(df_conditions.columns[0],cond1)
        ax2.text(text_label_wave,flux_match+0.2,text,color='k',alpha=0.8,fontsize=24,ha='left',va='bottom')

    ## format axes
    for ax in [ax1,ax2]:
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelsize=20,labeltop=True,labelbottom=True)
        ax.set_xlabel('Wavelength',fontsize=26)
        if ax==ax1:
            ax.set_ylim(0,y_offset_gap*len(condition1_sample)+0.6*y_offset_gap)
        if ax == ax2:
            ax.set_ylim(0,y_offset_gap*len(condition2_sample)+0.6*y_offset_gap)
        if log_x==True:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            # try to find the optimal major tick step
            desired_step = float('%s' % float('%.1g'%((wave[-1]-wave[0])/5)))
            major_steps = np.array([0.05,0.1,0.2,0.4,0.5,1,10,50,100,200,400,500,1000,1500,2000,3000,4000,5000])
            xaxis_major = major_steps[np.argmin(np.abs(major_steps-desired_step))]
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_major))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(xaxis_major/5))
        ax.grid(alpha=0.3)
        if log_y==True:
            ax.set_ylabel('$log_{10}(flux)$ + constant',fontsize=26)
        else:
            ax.set_ylabel('Normalized flux + constant',fontsize=26)
            
    return fig



## function to compare template with sample spectra
def comp_template_with_sample(df_buildingblock,df_sample_with_conditions,\
                              label_cols = ['cond1','cond2'],legend_label='sample',\
                              co_comp_Hsiao_temp = False,\
                              log_x=True,log_y=True,fig_ax=None,\
                              plot_gap = 0.7, ymax_shift = 0):
    """
    Input:
        df_buildingblock: pandas dataframe contains resulting PCA and GPR
        df_sample_with_conditions: pandas dataframe contains ['wave','flux','cond1','cond2'] in columns
        label_cols: default ['cond1','cond2'], the lables shows at the end of the each spectrum
        legend_label: default 'sample'
        co_comp_Hsiao_temp: if True, compare with the Hsiao template as well (Hsiao et al., 2007, 2009)
        log_x: default True, wavelength is plotted in log scale
        log_y: default True, flux is plotted in log scale
        fig_ax: if None, a new fig and ax will be created, if not None, then input [fig,ax]
        plot_gap: the yaxis-gap between each spectrum, default 0.5
        ymax_shift: the overall yaxis-shift, default 0
        
    Output:
        fig of template comparison with given sample
    """
    
    ## check if the columns needed are in the given sample dataframe
    required_columns = ['wave','flux','cond1','cond2']
    if np.sum([col in df_sample_with_conditions.columns for col in required_columns])<len(required_columns):
        raise ValueError('Please make sure the given comparison sample contains these columns: ["wave","flux","cond1","cond2"], and wavelength is in unit of A.')
        
    if co_comp_Hsiao_temp==True:
        print('Please make sure cond1 is epoch, cond2 is sBV if comparing with Hsiao template!')
    
    ## if the fig,ax already exist or not
    if fig_ax is None:
        fig, ax = plt.subplots(1,1,figsize=(10,15))
    else:
        fig, ax = fig_ax[0], fig_ax[1]
    
    ## plot the given sample and correspinding template (and co-compare template if needed)
    df_sample_with_conditions = df_sample_with_conditions.reset_index()
    for i,row in df_sample_with_conditions.iterrows():
        cond1,cond2 = row.cond1,row.cond2
        y_offset = ymax_shift + plot_gap*df_sample_with_conditions.shape[0]+1 - plot_gap*i
        ## get the template spectra 
        wv_temp,flux_temp,flux_err_temp = get_template(df_buildingblock,cond1,cond2,return_template_error=True)
        ## get the sample spectra at the same range
        wv_sample,fx_sample = row.wave.copy(),row.flux.copy() ## copy it so don't overwrite the df!!!!
        wv_range = [np.min(wv_temp)-(np.max(wv_temp)-np.min(wv_temp))/10,np.max(wv_temp)+(np.max(wv_temp)-np.min(wv_temp))/10]
        w = np.where((wv_sample>=wv_range[0])&(wv_sample<=wv_range[1]))
        wv_sample,fx_sample = wv_sample[w],fx_sample[w]
        ## plot them
        if log_y == True:
            plot_fx_sample = np.log10(fx_sample/np.mean(fx_sample))+y_offset
            plot_flux_temp = np.log10(flux_temp/np.mean(flux_temp))+y_offset
            plot_flux_temp_up_bond = np.log10((flux_temp+flux_err_temp)/np.mean(flux_temp))+y_offset
            plot_flux_temp_low_bond = np.log10((flux_temp-flux_err_temp)/np.mean(flux_temp))+y_offset
        else:
            plot_fx_sample = (fx_sample/np.mean(fx_sample))+y_offset
            plot_flux_temp = (flux_temp/np.mean(flux_temp))+y_offset
            plot_flux_temp_up_bond = ((flux_temp+flux_err_temp)/np.mean(flux_temp))+y_offset
            plot_flux_temp_low_bond = ((flux_temp-flux_err_temp)/np.mean(flux_temp))+y_offset
        if i==0: 
            sample_legend_label,temp_legend_label = legend_label, 'Template'
        else: 
            sample_legend_label,temp_legend_label = None,None
        ax.plot(wv_sample,plot_fx_sample,alpha=0.7,lw=1.6,color='grey',zorder=2,label=sample_legend_label)
        ax.plot(wv_temp,plot_flux_temp,alpha=0.7,lw=1.8,color='tab:purple',zorder=3)
        ax.fill_between(wv_temp,plot_flux_temp_low_bond,plot_flux_temp_up_bond,alpha=0.3,\
                        color='tab:purple',zorder=1,label=temp_legend_label)
        ax.text(wv_temp[-1]+(np.max(wv_temp)-np.min(wv_temp))/10,plot_flux_temp[-1],\
                ','.join(df_sample_with_conditions.loc[i,label_cols].astype(str)),fontsize=15,ha='left',va='center')
        
        ## compare with hsiao template if needed
        if co_comp_Hsiao_temp==True:
            stretched_epoch = cond1/cond2
            try:
                wv_hsiao,fx_hsiao = get_Hsiao_template(stretched_epoch)
                w = np.where((wv_hsiao>=wv_range[0])&(wv_hsiao<=wv_range[1]))
                wv_hsiao,fx_hsiao = wv_hsiao[w],fx_hsiao[w]
                if log_y == True:
                    plot_hsiao_fx = np.log10(fx_hsiao/np.mean(fx_hsiao))+y_offset
                else:
                    plot_hsiao_fx = fx_hsiao/np.mean(fx_hsiao)+y_offset
                if i==0:label='Hsiao template'
                else:label=None
                ax.plot(wv_hsiao,plot_hsiao_fx,alpha=0.8,color='tab:blue',lw=1.5,zorder=3,ls='--',label=label)
            except:
                continue
    
    ## other plot settings 
    ymin, ymax = 0, ymax_shift+ plot_gap*df_sample_with_conditions.shape[0]+2.8
    ax.set_xlim(wv_range[0],wv_range[1]+0.2*len(label_cols)*(np.max(wv_temp)-np.min(wv_temp)))
    ax.set_ylim(ymin,ymax)
    ax.grid(alpha=0.3)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=20,labeltop=True)
    ax.set_xlabel('Wavelength',fontsize=24)
    ax.legend(fontsize=20,loc='upper right',ncol=2+int(co_comp_Hsiao_temp))
    if log_x==True:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        # try to find the optimal major tick step
        desired_step = float('%s' % float('%.1g'%((wv_range[1]-wv_range[0])/5)))
        major_steps = np.array([0.05,0.1,0.2,0.4,0.5,1,10,50,100,200,300,400,500,1000,1500,2000,3000,4000,5000])
        xaxis_major = major_steps[np.argmin(np.abs(major_steps-desired_step))]
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xaxis_major/5))
    if log_y==True:
        ax.set_ylabel('$log_{10}(flux)$ + constant',fontsize=24)
    else:
        ax.set_ylabel('Normalized flux + constant',fontsize=24)
        
    return fig
        
    
