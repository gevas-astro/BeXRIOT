from uncertainties import ufloat,unumpy
from uncertainties.umath import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.stats import LombScargle
from scipy import signal
import warnings
warnings.filterwarnings("ignore")
import scipy.optimize
# from lmfit.models import GaussianModel
import glob
from astropy.table import Table,join,vstack,unique
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import wotan
import scipy.stats as st
import seaborn as sb
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

def sf(name,dpi=300,path='Figs/'):
    '''Save figure'''
    plt.savefig(path+name+'.png',dpi=dpi,bbox_inches='tight')

#initial LC, then color-mag, then a bunch of periodogram functions


def getIV(num,cross,printall=False,stack=False,both=True,plot=False,size=4,figsize=(8,4),zooms=False,mult=(3,40),offset=0,save=False,file='',radec=True,mlist=['OII I','OIII I'],calib=False):
    '''Uses table (cross) to make lists of I band and V band tables
    mult: tuple of multiples of orbital period to show
    offset: offset from beginning of light curve in days to use for zooms
    mlist: list of file names with masking in them; different for Part 2
    TO DO: add errors to plots'''
    #row of cross table using source number passed in
    crow = cross[cross['src_n']==num]
    #get RA and Dec to use in title
    if radec: ra,dec = crow['RA_OGLE'][0],crow['DEC_OGLE'][0]
    #get orbital period
    if crow['Porb'].mask[0]: orb_bool=False
    else:
        orb_bool=True
        orb = crow['Porb'][0]
    #list of I band tables (length <=3)
    iband = []
    for i in mlist:
        #doesn't work for OIV I since none are masked
        if not crow[i].mask[0]: #if there's a file (not masked)
            #read in table as temporary tab
            tab = Table.read(crow[i][0],format='ascii',names=['MJD-50000','I mag','I mag err'])
            #add tab to list of I band tables
            if len(tab)>0: iband.append(tab)
            else: print(f'empty file for {i}')
        else: 
            if printall: print('no file for '+i)
    #append OIV I band
    if len(mlist)<3:
        tab = Table.read(crow['OIV I'][0],format='ascii',names=['MJD-50000','I mag','I mag err'])
        if len(tab)>0: iband.append(tab)
        else: print(f'empty file for OIV I')
    
    #repeat for V band if both
    if both: 
        vband = []
        for v in ['OII V','OIII V','OIV V']:
            if not crow[v].mask[0]: #if there's a file (not masked)
                #read in table as temporary tab
                tab = Table.read(crow[v][0],format='ascii',names=['MJD-50000','V mag','V mag err'])
                #add tab to list of I band tables
                if len(tab)>0: vband.append(tab)
                else: print(f'empty file for {v}')
        #return lists of I band and V band tables
    #compensate for uncalibrated data by setting epochs to a common median, which is the overall median
    #V BAND UNCHANGED
    if calib:
        itemp = vstack(iband)
        med = np.median(itemp['I mag'])
        for i in iband:
            #calculate current median
            cmed = np.median(i['I mag'])
            #difference between current and target median
            dmeds = med-cmed
            #add difference to all points
            i['I mag'] += dmeds
    if plot:
        #stack for ease
        ib = vstack(iband)
        if both: vb = vstack(vband)

        #plot both full LC and two levels of zoom-in
        if zooms:
            #for now sharey but better to give different bounds for zooms 
            fig,[ax,ax1,ax2] = plt.subplots(3,1,figsize=figsize)         
        #plot both LCs
        else: fig,ax = plt.subplots(1,1,figsize=figsize)
        maxmag = 0
        minmag = np.inf
        ax.scatter(ib['MJD-50000'],ib['I mag'],color='#CF6275',s=size,label='I band')
        maxmag = np.max(ib['I mag'])
        minmag = np.min(ib['I mag'])
        if both:
            ax.scatter(vb['MJD-50000'],vb['V mag'],color='navy',s=size,label='V band')
            if np.max(vb['V mag'])>maxmag: 
                maxmag = np.max(vb['V mag'])
            if np.min(vb['V mag'])<minmag: 
                minmag = np.min(vb['V mag'])
        ax.set_xlabel('MJD-50000',fontsize=14)
        ax.set_ylabel('OGLE mag',fontsize=14)
        ax.set_ylim(maxmag+.05,minmag-.05)
        if radec: ax.set_title('Source #'+str(num)+' RA: '+str(ra)+' Dec: '+str(dec))
        else: ax.set_title('Source #'+str(num))
        ax.legend()
        if zooms: #for now just plots I band
            #ax1 zoom is hundreds of days
            #find median time spacing between points
            samp = np.median(ib['MJD-50000'][1:] - ib['MJD-50000'][:-1])
            if orb_bool:
                inds1 = int(mult[1]*orb/samp)
                inds2 = int(mult[0]*orb/samp)
                start = int(offset/samp)
                zi1 = ib[start:start+inds1]
                zi2 = ib[start:start+inds2]
                ax1.scatter(zi1['MJD-50000'],zi1['I mag'],color='#CF6275',s=size+4)
                ax2.scatter(zi2['MJD-50000'],zi2['I mag'],color='#CF6275',s=size+4)
                #find min and max for each and adjust y lim
                max1,min1 = np.max(zi1['I mag']),np.min(zi1['I mag'])
                max2,min2 = np.max(zi2['I mag']),np.min(zi2['I mag'])
                ax1.set_ylim(max1+.02,min1-.02)
                ax2.set_ylim(max2+.02,min2-.02)               
                print('orbital period:',orb)

            else: #if no known orbital period
                #TO DO add in offset use here
                inds1 = int(1000/samp)
                inds2 = int(100/samp)
                zi1 = ib[:inds1]
                zi2 = ib[:inds2]
                ax1.scatter(zi1['MJD-50000'],zi1['I mag'],color='#CF6275',s=size+4)
                ax2.scatter(zi2['MJD-50000'],zi2['I mag'],color='#CF6275',s=size+4)
                #find min and max for each and adjust y lim
                max1,min1 = np.max(zi1['I mag']),np.min(zi1['I mag'])
                max2,min2 = np.max(zi2['I mag']),np.min(zi2['I mag'])
                ax1.set_ylim(max1+.02,min1-.02)
                ax2.set_ylim(max2+.02,min2-.02)  
    if save: plt.savefig(file+'.png',dpi=200,bbox_inches='tight')
    if stack and both: return vstack(iband),vstack(vband)
    elif both: return iband,vband
    elif stack: return vstack(iband)
    else: return iband


def colormag(iband,vband,figsize=(7,8),plot=True,printcorr=True,retint=False,ctime=True,cmap='viridis',both=True,save=False,file=''):
    '''Interpolates I band data at times of V band and then plots color-mag with best fit and corr coeff.
    Now assumes iband and vband are single tables, but can add option to vstack in function if needed.'''
    #interpolate I band
    i_interp = np.interp(vband['MJD-50000'],iband['MJD-50000'],iband['I mag'])
    
    if plot:
        if both: 
            fig,(ax,ax1) = plt.subplots(2,1,figsize=figsize,sharex=True)
            plt.subplots_adjust(hspace=0.05)
            axlist = [ax,ax1]
        else: 
            fig,ax = plt.subplots(1,1,figsize=figsize)
            axlist = [ax]
        #approximate interpolated I errors as median I band error (was using max but could be issue w/outliers)
        ie = np.ones(len(i_interp))*np.median(iband['I mag err'])
        #propagate errors to get error on V-I points
        verr = unumpy.uarray(vband['V mag'],vband['V mag err'])
        ierr = unumpy.uarray(i_interp,ie)
        v_i = verr-ierr
        #just take errors
        v_i_err = unumpy.std_devs(v_i)
        
        #plot Iint vs. V-I
        if ctime:
            im = ax.scatter(vband['V mag']-i_interp,i_interp,c=vband['MJD-50000'],cmap=cmap,zorder=10)
            #add errorbars
            ax.errorbar(vband['V mag']-i_interp,i_interp,yerr=ie,xerr=v_i_err,color='grey',zorder=0,ls='none',marker='')
            if both: 
                ax1.scatter(vband['V mag']-i_interp,vband['V mag'],c=vband['MJD-50000'],cmap=cmap,zorder=10)
                #add errorbars separately
                ax1.errorbar(vband['V mag']-i_interp,vband['V mag'],yerr=vband['V mag err'],xerr=v_i_err,color='grey',zorder=0,ls='none',marker='')
            fig.colorbar(im, ax=axlist,label='MJD-50000')        
        else: 
            ax.errorbar(vband['V mag']-i_interp,i_interp,yerr=ie,xerr=v_i_err,color='black',linestyle='none',marker='o')
            if both: ax1.errorbar(vband['V mag']-i_interp,vband['V mag'],yerr=vband['V mag err'],xerr=v_i_err,color='black',linestyle='none',marker='o')
        #flip y-axis such that positive corr on plot is redder when brighter
        maxi,mini = np.max(i_interp),np.min(i_interp)
        maxv,minv = np.max(vband['V mag']),np.min(vband['V mag'])
        
        
        ax.set_ylim(maxi+.04,mini-.04)
        if both:ax1.set_ylim(maxv+.04,minv-.04)
        
        ax.set_ylabel(r'$\mathrm{I_{int}}$',fontsize=14)
        if both: 
            ax1.set_xlabel(r'$\mathrm{V - I_{int}}$',fontsize=14)
            ax1.set_ylabel('V',fontsize=14)
        else: ax.set_xlabel(r'$\mathrm{V - I_{int}}$',fontsize=14)
    if printcorr:
        #print correlation corr with interpolated I and V-I and then V and V-I
        print('I and V-I correlation:',np.corrcoef(vband['V mag']-i_interp,i_interp)[1][0])
        print('V and V-I correlation:',np.corrcoef(vband['V mag']-i_interp,vband['V mag'])[1][0])
    if save: plt.savefig(file+'.png',dpi=200,bbox_inches='tight')
    if retint or not plot: return i_interp
    else: return


def two_sided_gaussian(t, A, t0, sigma_left, sigma_right):
    return np.where(t < t0,
                    A * np.exp(-(t - t0)**2 / (2 * sigma_left**2)),
                    A * np.exp(-(t - t0)**2 / (2 * sigma_right**2)))

def background_polynomial(t, a0, a1, a2):
    return a0 + a1 * t + a2 * t**2

def model_function(t, a0, a1, a2, A, t0, sigma_left, sigma_right):
    polynomial = a0 + a1 * (t-np.min(t)) + a2 * (t-np.min(t))**2
    flare = -two_sided_gaussian(t, A, t0, sigma_left, sigma_right)
    return polynomial + flare

def fit_time_series(time, mag, err_mag, initial_t0):
    # Initial guesses for the parameters
    # p0 = [0, 0, 0, 1, initial_t0, 1, 1]  # [a0, a1, a2, A, t0, sigma_left, sigma_right]
    p0 = [np.mean(mag), 0, 0, max(mag) - min(mag), initial_t0, 30, 30]
    
    lower_bounds = [0, -np.inf, -np.inf, 0, initial_t0 - 300,1, 5]  # Lower bounds
    upper_bounds = [20, np.inf, np.inf, 3, initial_t0 + 300, np.inf, np.inf]  # Upper bounds
    
    # #best for J0520 flat BG
    # lower_bounds = [0, -1E-10, -1E-10, 0, initial_t0 - 300,1, 5]  # Lower bounds
    # upper_bounds = [20, 1E-10, 1E-10, 3, initial_t0 + 300, np.inf, np.inf]  # Upper bounds
    
    # Fit the model to the data
    popt, pcov = curve_fit(model_function, time, mag, sigma=err_mag, p0=p0,bounds=(lower_bounds, upper_bounds))
    # popt, pcov = curve_fit(model_function, time, mag, sigma=err_mag, p0=p0)
    
    # Residuals
    residuals = mag - model_function(time, *popt)

    # Chi-square
    chi2 = np.sum((residuals / (err_mag)) ** 2)
    # chi2 = np.sum(residuals**2 / mag)
    dof = len(mag) - len(popt)  # Degrees of freedom
    chi2_red = chi2 / dof
    # print(err_mag,chi2)
    print(np.min(err_mag))
    print(f"Chi-squared: {chi2}, Reduced Chi-squared: {chi2_red:.2f}, DOF: {len(mag):.2f}/{len(popt):.2f}")

    return popt, pcov


def ozoom(iband,mjd1,mjd2):
    mask_I = (iband['MJD-50000'] >mjd1) & (iband['MJD-50000'] <mjd2) 
    t_I = iband['MJD-50000'][mask_I]
    m_I = iband['I mag'][mask_I]
    em_I = iband['I mag err'][mask_I]
    plt.plot(t_I,m_I,'.')
    plt.ylim(np.max(m_I)+0.05,np.min(m_I)-0.05)
    
    
def ozoom_fit(iband,mjd1,mjd2):
    mask_I = (iband['MJD-50000'] >mjd1) & (iband['MJD-50000'] <mjd2) 
    time = iband['MJD-50000'][mask_I]
    mag = iband['I mag'][mask_I]
    err_mag = iband['I mag err'][mask_I]
    
    initial_t0 = 0.5*(mjd1 + mjd2)  # Your initial guess for the flare peak time

    popt, pcov = fit_time_series(time, mag, err_mag, initial_t0)
    
    # Extract the diagonal of the covariance matrix to get the variances
    perr = np.sqrt(np.diag(pcov))

    # Print the best-fit parameters and their 1-sigma uncertainties
    print(f"Amplitude (A): {A:.2f} ± {perr[3]:.2f}")
    print(f"Peak time (t0): {t0:.2f} ± {perr[4]:.2f}")
    print(f"Sigma (left): {sigma_left:.2f} ± {perr[5]:.2f}")
    print(f"Sigma (right): {sigma_right:.2f} ± {perr[6]:.2f}")
    
    print("Fitted parameters:", popt)
    
    # Extract the optimized parameters
    a0, a1, a2, A, t0, sigma_left, sigma_right = popt
    
    tmod = np.linspace(np.min(time),np.max(time),1000)
    ymodel = model_function(tmod, *popt)
    # polynomial_comp = a0 + a1 * (tmod-np.min(time)) + a2 * (tmod-np.min(time))**2
    polynomial_comp = background_polynomial(tmod-np.min(time), a0, a1, a2)
    flare_comp = -two_sided_gaussian(tmod, A, t0, sigma_left, sigma_right)  # Negative for magnitude
    
    min_mag_index = np.argmin(mag)
    brightest_time = time[min_mag_index]
    brightest_mag = mag[min_mag_index]
    
    background_mag = background_polynomial(brightest_time-np.min(time), a0, a1, a2)

    background_magA = background_polynomial(t0-np.min(time), a0, a1, a2)
    # Calculate the magnitude difference
    magnitude_difference = background_mag -brightest_mag
    
    
    # Adding the vertical line with arrows
    # plt.vlines(
    #     x=brightest_time, 
    #     ymin=background_mag, 
    #     ymax=brightest_mag, 
    #     colors='purple', 
    #     linestyles='dotted', 
    #     label='Flare strength'
    # )
    
    plt.figure(figsize=(8,4))
    
    plt.annotate('', xy=(brightest_time, brightest_mag), xytext=(brightest_time, background_mag),
                 arrowprops=dict(arrowstyle='<->', color='purple',ls=':'))
    
    text_x = brightest_time - 60  # 20 days offset to the left of the peak
    text_y = brightest_mag  # y-position at the brightest magnitude point
    plt.text(text_x, text_y, f"Flare Δmag: {magnitude_difference:.2f}", 
         horizontalalignment='right', verticalalignment='center', color='purple')
    
    text_y2 = brightest_mag + 0.05  # Adjust y-position slightly above the first line
    plt.text(text_x, text_y2, f"Fit Amplitude: {A:.2f}", 
         horizontalalignment='right', verticalalignment='center', color='green')

    
    # plt.axhline(background_mag)
    
    plt.errorbar(time, mag, yerr=err_mag, fmt='.', label='Data', color='black')
    plt.plot(tmod, ymodel, label='Best fit', color='blue')
    plt.plot(tmod, polynomial_comp, label='BG poly.', color='red', linestyle='--')
    plt.plot(tmod, np.max(mag)+flare_comp, label='Flare', color='green', linestyle='-.')
    plt.xlabel('MJD -50000')
    plt.ylabel('OGLE I mag')
    plt.gca().invert_yaxis()  # Invert y-axis for magnitude plot
    plt.legend()
    plt.title('Light Curve Fitting with Flare and Background')
    plt.ylim(np.max(mag)+0.01,np.min(mag)-0.05)
    plt.xlim(np.min(time)-5,np.max(time)+5)
    print('Peak intensity: ',A, magnitude_difference)
    
    return A


def ozoom_fit2(iband, mjd1, mjd2,title='source'):
    mask_I = (iband['MJD-50000'] > mjd1) & (iband['MJD-50000'] < mjd2)
    time = iband['MJD-50000'][mask_I]
    mag = iband['I mag'][mask_I]
    err_mag = iband['I mag err'][mask_I]

    # Initial guess for the flare peak time
    initial_t0 = 0.5 * (mjd1 + mjd2)

    # Perform the fit
    popt, pcov = fit_time_series(time, mag, err_mag, initial_t0)
    
    # Extract the optimized parameters
    a0, a1, a2, A, t0, sigma_left, sigma_right = popt
    
    # Extract the diagonal of the covariance matrix to get the variances
    perr = np.sqrt(np.diag(pcov))

    # Print the best-fit parameters and their 1-sigma uncertainties
    print(f"Amplitude (A): {A:.3f} ± {perr[3]:.3f}")
    print(f"Peak time (t0): {t0:.2f} ± {perr[4]:.2f}")
    print(f"Sigma (left): {sigma_left:.2f} ± {perr[5]:.2f}")
    print(f"Sigma (right): {sigma_right:.2f} ± {perr[6]:.2f}")
    
    print("Fitted parameters:", popt)
    

    tmod = np.linspace(np.min(time), np.max(time), 1000)
    ymodel = model_function(tmod, *popt)
    polynomial_comp = background_polynomial(tmod - np.min(time), a0, a1, a2)
    flare_comp = -two_sided_gaussian(tmod, A, t0, sigma_left, sigma_right)  # Negative for magnitude

    # Calculate the magnitude difference
    min_mag_index = np.argmin(mag)
    brightest_time = time[min_mag_index]
    brightest_mag = mag[min_mag_index]
    background_mag = background_polynomial(brightest_time - np.min(time), a0, a1, a2)
    magnitude_difference = background_mag - brightest_mag
    
    background_magA = background_polynomial(t0-np.min(time), a0, a1, a2)

    # Create a figure with two panels (subplots)
    fig = plt.figure(figsize=(6, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])  # Adjust height ratios as needed

    # Upper panel (entire time series)
    ax0 = plt.subplot(gs[0])
    ax0.errorbar(iband['MJD-50000'], iband['I mag'], yerr=iband['I mag err'], fmt='.', color='black', label='Data')
    ax0.invert_yaxis()
    ax0.set_ylabel('OGLE I mag')
    ax0.set_title(title)
    ax0.legend()

    # Draw a rectangle indicating the zoomed-in region
    zoom_rect = patches.Rectangle((mjd1, np.min(mag) - 0.5), mjd2 - mjd1, np.ptp(mag) + 0.5, linewidth=1,
                                  edgecolor='red', facecolor='none')
    ax0.add_patch(zoom_rect)

    # Lower panel (zoomed-in)
    ax1 = plt.subplot(gs[1])
    ax1.errorbar(time, mag, yerr=err_mag, fmt='.', label='Data', color='black')
    ax1.plot(tmod, ymodel, label='Best fit', color='blue')
    ax1.plot(tmod, polynomial_comp, label='BG poly.', color='red', linestyle='--')
    ax1.plot(tmod, np.max(mag) + flare_comp, label='Flare', color='green', linestyle='-.')

    ax1.annotate('', xy=(brightest_time, brightest_mag), xytext=(brightest_time, background_mag),
                 arrowprops=dict(arrowstyle='<->', color='purple', ls=':'))

    text_x = brightest_time - 60  # Adjust as needed
    text_y = brightest_mag  # y-position at the brightest magnitude point
    ax1.text(text_x, text_y, f"Flare Δmag: {magnitude_difference:.2f}",
             horizontalalignment='right', verticalalignment='center', color='purple')

    text_y2 = brightest_mag + 0.05  # Adjust y-position slightly above the first line
    ax1.text(text_x, text_y2, f"Fit Amplitude: {A:.2f}",
             horizontalalignment='right', verticalalignment='center', color='green')

    ax1.set_xlabel('MJD -50000')
    ax1.set_ylabel('OGLE I mag')
    ax1.invert_yaxis()
    ax1.set_xlim(np.min(time) - 5, np.max(time) + 5)
    ax1.set_ylim(np.max(mag) + 0.01, np.min(mag) - 0.05)
    ax1.legend()
    # ax1.set_title('Zoomed-In Light Curve Fitting')

    plt.tight_layout()
    plt.minorticks_on()
    ax0.minorticks_on()
    ax1.minorticks_on()
    title2=title.replace(" ", "_")
    sf(title2+'_v1')
    # plt.show()

    print('Peak intensity:', background_mag, magnitude_difference)
    print('Peak intensity:', background_magA, A)
    return A


def ozoom_fit3(iband,datax, mjd1, mjd2,title='source',figsize=(6, 4.5)):
    scale=50000
    mjdx = datax[0]
    ratex = datax[3]
    ratepos = datax[4]
    rateneg = -datax[5]
    mask1x = (ratepos > 0) & (mjdx <58800)
    mask2x = (ratepos == 0) & (mjdx <58800)
    
    mask_I = (iband['MJD-50000'] > mjd1) & (iband['MJD-50000'] < mjd2)
    time = iband['MJD-50000'][mask_I]
    mag = iband['I mag'][mask_I]
    err_mag = iband['I mag err'][mask_I]

    # Initial guess for the flare peak time
    initial_t0 = 0.5 * (mjd1 + mjd2)

    # Perform the fit
    popt, pcov = fit_time_series(time, mag, err_mag, initial_t0)
    
    # Extract the optimized parameters
    a0, a1, a2, A, t0, sigma_left, sigma_right = popt
    
    
    
    # Extract the diagonal of the covariance matrix to get the variances
    perr = np.sqrt(np.diag(pcov))

    # Print the best-fit parameters and their 1-sigma uncertainties
    print(f"Amplitude (A): {A:.3f} ± {perr[3]:.3f}")
    print(f"Peak time (t0): {t0:.2f} ± {perr[4]:.2f}")
    print(f"Sigma (left): {sigma_left:.2f} ± {perr[5]:.2f}")
    print(f"Sigma (right): {sigma_right:.2f} ± {perr[6]:.2f}")
    
    print("Fitted parameters:", popt)
    

    tmod = np.linspace(np.min(time), np.max(time), 1000)
    ymodel = model_function(tmod, *popt)
    polynomial_comp = background_polynomial(tmod - np.min(time), a0, a1, a2)
    flare_comp = -two_sided_gaussian(tmod, A, t0, sigma_left, sigma_right)  # Negative for magnitude

    # Calculate the magnitude difference
    min_mag_index = np.argmin(mag)
    brightest_time = time[min_mag_index]
    brightest_mag = mag[min_mag_index]
    background_mag = background_polynomial(brightest_time - np.min(time), a0, a1, a2)
    magnitude_difference = background_mag - brightest_mag
    
    background_magA = background_polynomial(t0-np.min(time), a0, a1, a2)

    # Create a figure with two panels (subplots)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1,1, 3])  # Adjust height ratios as needed
    plt.subplots_adjust(hspace=0.0)

    # Upper panel (entire time series)
    ax0 = plt.subplot(gs[0])
    # ax0.errorbar(iband['MJD-50000'], iband['I mag'], yerr=iband['I mag err'], fmt='.', color='black', label='Data')
    ax0.errorbar(iband['MJD-50000'], iband['I mag'], yerr=iband['I mag err'],fmt=' ', color='black', label='Data')
    ax0.invert_yaxis()
    ax0.set_ylabel('OGLE I mag')
    ax0.set_title(title)
    # ax0.legend()

    # Draw a rectangle indicating the zoomed-in region
    zoom_rect = patches.Rectangle((mjd1, np.min(mag) - 0.5), mjd2 - mjd1, np.ptp(mag) + 0.5, linewidth=1,
                                  edgecolor='red', facecolor='none')
    ax0.add_patch(zoom_rect)
    
    ax1 = plt.subplot(gs[1],sharex=ax0)
    ax1.errorbar(mjdx[mask1x]-scale, ratex[mask1x], yerr=[rateneg[mask1x], ratepos[mask1x]], fmt=' ', label='XRT', capsize=1, color='black', alpha=0.5, zorder=-10)
    ax1.scatter(mjdx[mask2x]-scale, ratex[mask2x], c='grey',marker=r'$\downarrow$',s=20, label='XRT up-lim', alpha=0.2 )
    ax1.set_ylabel('XRT (c/s)')
    # ax1.set_xlabel('MJD -50000')
    # ax1.set_yscale('log')
    ax1.legend()

    # Lower panel (zoomed-in)
    ax2 = plt.subplot(gs[2])
    ax2.errorbar(time, mag, yerr=err_mag, fmt='.', label='Data', color='black')
    ax2.plot(tmod, ymodel, label='Best fit', color='blue')
    ax2.plot(tmod, polynomial_comp, label='BG poly.', color='red', linestyle='--')
    ax2.plot(tmod, np.max(mag) + flare_comp, label='Flare', color='green', linestyle='-.')

    ax2.annotate('', xy=(brightest_time, brightest_mag), xytext=(brightest_time, background_mag),
                 arrowprops=dict(arrowstyle='<->', color='purple', ls=':'))

    text_x = brightest_time - 60  # Adjust as needed
    text_y = brightest_mag  # y-position at the brightest magnitude point
    ax2.text(text_x, text_y, f"Flare Δmag: {magnitude_difference:.2f}",
             horizontalalignment='right', verticalalignment='center', color='purple')

    text_y2 = brightest_mag + 0.05  # Adjust y-position slightly above the first line
    ax2.text(text_x, text_y2, f"Fit Amplitude: {A:.2f}",
             horizontalalignment='right', verticalalignment='center', color='green')

    ax2.set_xlabel('MJD -50000')
    ax2.set_ylabel('OGLE I mag')
    ax2.invert_yaxis()
    ax2.set_xlim(np.min(time) - 5, np.max(time) + 5)
    ax2.set_ylim(np.max(mag) + 0.01, np.min(mag) - 0.05)
    ax2.legend(loc='upper right')
    # ax1.set_title('Zoomed-In Light Curve Fitting')

    ax0.tick_params(labelbottom=False)
    # ax0.spines['bottom'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.minorticks_on()
    ax0.minorticks_on()
    ax1.minorticks_on()
    ax2.minorticks_on()
    title2=title.replace(" ", "_")
    # plt.show()
    
     # Adjust panel positions manually
    # fig.subplots_adjust(hspace=0)  # Set zero spacing between all subplots
    ax0_pos = ax0.get_position()  # Get the original position of ax0
    ax1_pos = ax1.get_position()  # Get the original position of ax1
    ax2_pos = ax2.get_position()  # Get the original position of ax2

#     # Modify positions to make ax0 and ax1 touch
    ax0.set_position([ax0_pos.x0, ax0_pos.y0 - 0.03, ax0_pos.width, ax0_pos.height+0.03])
    ax1.set_position([ax1_pos.x0, ax1_pos.y0 - 0.00, ax1_pos.width, ax1_pos.height+0.03])
    ax2.set_position([ax2_pos.x0, ax2_pos.y0, ax2_pos.width, ax2_pos.height])
    
    sf(title2+'_v3')
    
    # ax0.axvline(6700)
    # ax1.axvline(6700)
    # ax2.axvline(6700)



    # Save and display the plot
    # plt.tight_layout()

    print('Peak intensity:', background_mag, magnitude_difference)
    print('Peak intensity:', background_magA, A)
    return A


