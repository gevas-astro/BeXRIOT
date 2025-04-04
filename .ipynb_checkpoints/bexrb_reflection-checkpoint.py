import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm
# from scipy.constants import h, c, k
from astropy import units as u
from astropy import constants as const

import pandas as pd
from scipy.interpolate import CubicSpline
from scipy import integrate

sigma_T=(const.sigma_T).cgs.value
sigma_SB=(const.sigma_sb).cgs.value
sigma_wien = 0.2897755
evtoerg = 1.6e-12
A_to_cm = (u.angstrom).to(u.cm)
kpc_to_cm = 3.086e21
c=(const.c).cgs.value
k=(const.k_B).cgs.value
h=(const.h).cgs.value
Rsun = 6.96E+10
sigma = (const.sigma_sb).cgs.value  # Stefan-Boltzmann constant in erg cm^-2 s^-1 K^-4

def sf(name,dpi=300,path='Figs/'):
    '''Save figure'''
    plt.savefig(path+name+'.png',dpi=dpi,bbox_inches='tight')

def temp_Carciofi(r, R_star, T_star):
    #  A. C. Carciofi and J. E. Bjorkman 2006 ApJ 639 1081
    # aprox 0.75 *T0_star * (R_star / r) ** 0.8
    term1 = np.arcsin(R_star / r)
    term2 = (R_star / r) * np.sqrt(1 - (R_star / r) ** 2)
    return (T_star / np.pi**0.25) * (term1 - term2) ** 0.25
    
def compute_temperature_change_cgs_disk(radius_star, num_points, point_source_distance, T0_star, L, inner_disk_radius, outer_disk_radius, inclination_angle_y, inclination_angle_z, T0_disk, albedo=0.5, n=2, Carciofi=True):

    # Create a grid of points representing the star
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    u, v = np.meshgrid(u, v)
    
    x_star = radius_star * np.cos(u) * np.sin(v)
    y_star = radius_star * np.sin(u) * np.sin(v)
    z_star = radius_star * np.cos(v)
    
    # Coordinates of the point source
    point_source = np.array([point_source_distance, 0, 0])
    
    # Compute distances and cosines for the star
    distances_star = np.sqrt((x_star - point_source[0])**2 + (y_star - point_source[1])**2 + (z_star - point_source[2])**2)
    normal_vectors_star = np.array([x_star, y_star, z_star])
    direction_vectors_star = np.array([point_source[0] - x_star, point_source[1] - y_star, point_source[2] - z_star])
    dot_products_star = np.einsum('ijk,ijk->jk', normal_vectors_star, direction_vectors_star)
    magnitudes_star = np.sqrt(np.sum(normal_vectors_star**2, axis=0)) * distances_star
    cos_phi_star = np.maximum(dot_products_star / magnitudes_star, 0)
    
    # Temperature change for the star
    T0_4_star = T0_star**4
    temperature_change_star = (1 - albedo) * L * cos_phi_star / (4 * np.pi * sigma * distances_star**2)
    T_new_star = (T0_4_star + temperature_change_star)**0.25
    
    # Mask for areas where cos_phi < 0 (unilluminated)
    mask_star = (cos_phi_star < 0)
    T_new_star[mask_star] = T0_star
    
    # Create a grid of points representing the disk
    r_disk = np.linspace(inner_disk_radius, outer_disk_radius, num_points)
    theta_disk = np.linspace(0, 2 * np.pi, num_points)
    r_disk, theta_disk = np.meshgrid(r_disk, theta_disk)
    
    x_disk = r_disk * np.cos(theta_disk)
    y_disk = r_disk * np.sin(theta_disk)
    z_disk = np.zeros_like(x_disk)  # Initially flat disk in the x-y plane

    
    # Apply inclination around the y-axis
    inclination_y_rad = np.deg2rad(inclination_angle_y)
    x_rotated_y = x_disk * np.cos(inclination_y_rad) + z_disk * np.sin(inclination_y_rad)
    y_rotated_y = y_disk
    z_rotated_y = -x_disk * np.sin(inclination_y_rad) + z_disk * np.cos(inclination_y_rad)

    # Apply inclination around the z-axis
    inclination_z_rad = np.deg2rad(inclination_angle_z)
    x_disk_inclined = x_rotated_y * np.cos(inclination_z_rad) - y_rotated_y * np.sin(inclination_z_rad)
    y_disk_inclined = x_rotated_y * np.sin(inclination_z_rad) + y_rotated_y * np.cos(inclination_z_rad)
    z_disk_inclined = z_rotated_y  # z remains unchanged for rotation around z-axis
    
    # Compute distances and cosines for the disk
    distances_disk = np.sqrt((x_disk_inclined - point_source[0])**2 + (y_disk_inclined - point_source[1])**2 + (z_disk_inclined - point_source[2])**2)

    # Normal vector initially pointing in z-direction
    normal_vectors_disk = np.array([0, 0, 1])

    # Apply y-axis rotation to the normal vector
    normal_vector_rotated_y = np.array([
        normal_vectors_disk[0] * np.cos(inclination_y_rad) + normal_vectors_disk[2] * np.sin(inclination_y_rad),
        normal_vectors_disk[1],
        -normal_vectors_disk[0] * np.sin(inclination_y_rad) + normal_vectors_disk[2] * np.cos(inclination_y_rad)
    ])

    # Apply z-axis rotation to the normal vector
    normal_vector_inclined = np.array([
        normal_vector_rotated_y[0] * np.cos(inclination_z_rad) - normal_vector_rotated_y[1] * np.sin(inclination_z_rad),
        normal_vector_rotated_y[0] * np.sin(inclination_z_rad) + normal_vector_rotated_y[1] * np.cos(inclination_z_rad),
        normal_vector_rotated_y[2]
    ])

    # Normalize the normal vector
    normal_vector_inclined /= np.linalg.norm(normal_vector_inclined)

    # Direction vectors from the point source to each point on the disk
    direction_vectors_disk = np.array([point_source[0] - x_disk_inclined, point_source[1] - y_disk_inclined, point_source[2] - z_disk_inclined])

    # Compute dot products of normal vectors and direction vectors
    dot_products_disk = (normal_vector_inclined[0] * direction_vectors_disk[0] +
                         normal_vector_inclined[1] * direction_vectors_disk[1] +
                         normal_vector_inclined[2] * direction_vectors_disk[2])

    # Compute cos(phi) for the disk
    magnitudes_disk = distances_disk
    cos_phi_disk = np.abs(dot_products_disk / magnitudes_disk)



    # Apply the radial temperature profile
    radial_distances = np.sqrt(x_disk**2 + y_disk**2)
    # T_profile_disk = T0_disk * (radial_distances / radius_star)**(-n) #simple profile.
    if Carciofi:
        print('compute Carciofi')
        T_profile_disk = temp_Carciofi(radial_distances, radius_star, T0_star)# T0_disk should be the same as star for this expresion.
    else:
        print('compute pow')
        T_profile_disk = T0_disk * (radial_distances / radius_star)**(-n)

    # Temperature change for the disk including the radial profile
    T0_4_disk = T_profile_disk**4
    temperature_change_disk = (1 - albedo) * L * cos_phi_disk / (4 * np.pi * sigma * distances_disk**2)
    T_new_disk = (T0_4_disk + temperature_change_disk)**0.25
    
    return (x_star, y_star, z_star, T_new_star, point_source, x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk)


def plot_star_and_disk_with_temperature_change(radius_star, num_points, point_source_distance, T0_star, L, inner_disk_radius, outer_disk_radius, inclination_angle_y, inclination_angle_z, T0_disk, albedo=0.5, n=2,flag=False,Carciofi=True):
    # Compute temperature change (assuming compute function is implemented correctly)
    x_star, y_star, z_star, T_new_star, point_source, x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk = compute_temperature_change_cgs_disk(
        radius_star, num_points, point_source_distance, T0_star, L, inner_disk_radius, outer_disk_radius, inclination_angle_y, inclination_angle_z, T0_disk, albedo=albedo, n=n, Carciofi=Carciofi
    )

    # Normalize temperatures for colormap
    norm_star = (T_new_star - T0_star - T_new_star.min()) / (T_new_star.max() - T_new_star.min())
    # norm_disk = (T_new_disk - T0_disk - T_new_disk.min()) / (T_new_disk.max() - T_new_disk.min())
    norm_disk = (T_new_disk - T_new_disk.min()) / (T_new_disk.max() - T_new_disk.min())
    surfacecolor_disk=T_new_disk -T_new_disk.min()
    title_disk='ΔT Disk (K)'
    if flag:
        surfacecolor_disk = T_new_disk
        title_disk = 'T Disk (K)'

    colors_star = plt.cm.viridis(norm_star)[:, :3]  # Extract RGB from colormap
    colors_disk = plt.cm.inferno(norm_disk)[:, :3]  # Extract RGB from colormap

    # Create Plotly figure
    fig = go.Figure()

    # Plot the star surface with the temperature change
    fig.add_trace(go.Surface(
        x=x_star/Rsun,
        y=y_star/Rsun,
        z=z_star/Rsun,
        surfacecolor=T_new_star - T0_star,
        colorscale='Viridis',
        colorbar=dict(
            # title=r'$\Delta T\ Star\ (K)$',
            title='ΔT Star (K)',
            titleside='right',
            tickprefix='',
            len=0.5,  # Adjust the length of the color bar
            xanchor='right',
            yanchor='middle',
            x=1.05,  # Position color bar outside the plot
            y=0.5
        ),
        opacity=1.0,  # Fully opaque
        name='Star Surface'
    ))

    # Plot the disk surface with the temperature change
    fig.add_trace(go.Surface(
        x=x_disk_inclined/Rsun,
        y=y_disk_inclined/Rsun,
        z=z_disk_inclined/Rsun,
        # surfacecolor=T_new_disk - T0_disk,
        surfacecolor=surfacecolor_disk,
        colorscale='inferno',
        colorbar=dict(
            title = title_disk,
            titleside ='right',
            tickprefix ='',
            len=0.5,  # Adjust the length of the color bar
            xanchor='right',
            yanchor='middle',
            x=1.25,  # Position color bar outside the plot
            y=0.5
        ),
        opacity=1.0,  # Fully opaque
        name='Disk Surface'
    ))

    # Add the illuminating source as a small sphere
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    u, v = np.meshgrid(u, v)
    small_radius = radius_star * 0.2
    x_source = small_radius * np.cos(u) * np.sin(v) + point_source[0]
    y_source = small_radius * np.sin(u) * np.sin(v)
    z_source = small_radius * np.cos(v)
    
    fig.add_trace(go.Surface(
        x=x_source/Rsun,
        y=y_source/Rsun,
        z=z_source/Rsun,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        showscale=False,  # Hide color bar for this surface
        opacity=1.0,  # Fully opaque
        name='Point Source',
        showlegend=False  # Hide this trace from the legend
    ))


    # Plot the line from the point source to the closest point on the disk
    # vector_to_source = point_source
    # inclination_rad = np.deg2rad(inclination_angle)
    # normal_vector = np.array([np.sin(inclination_rad), 0, np.cos(inclination_rad)])
    # distance_to_plane = np.dot(vector_to_source, normal_vector)
    # closest_point_on_disk = point_source - distance_to_plane * normal_vector
    
    # Compute the normal vector to the inclined disk
    inclination_y_rad = np.deg2rad(inclination_angle_y)
    inclination_z_rad = np.deg2rad(inclination_angle_z)

    # Initially, the normal vector is along the z-axis (0, 0, 1)
    normal_vector_initial = np.array([0, 0, 1])

    # Apply rotation around the y-axis
    normal_vector_y_rotated = np.array([
        normal_vector_initial[0] * np.cos(inclination_y_rad) + normal_vector_initial[2] * np.sin(inclination_y_rad),
        normal_vector_initial[1],
        -normal_vector_initial[0] * np.sin(inclination_y_rad) + normal_vector_initial[2] * np.cos(inclination_y_rad)
    ])

    # Apply rotation around the z-axis
    normal_vector_inclined = np.array([
        normal_vector_y_rotated[0] * np.cos(inclination_z_rad) - normal_vector_y_rotated[1] * np.sin(inclination_z_rad),
        normal_vector_y_rotated[0] * np.sin(inclination_z_rad) + normal_vector_y_rotated[1] * np.cos(inclination_z_rad),
        normal_vector_y_rotated[2]
    ])

    # Normalize the normal vector
    normal_vector_inclined /= np.linalg.norm(normal_vector_inclined)

    # Vector from the point source to a point on the plane (origin)
    vector_to_source = point_source - np.array([0, 0, 0])

    # Distance to the plane along the normal vector
    distance_to_plane = np.dot(vector_to_source, normal_vector_inclined)

    # Project the point source onto the plane
    closest_point_on_disk = point_source - distance_to_plane * normal_vector_inclined


    fig.add_trace(go.Scatter3d(
        x=[point_source[0]/Rsun, closest_point_on_disk[0]/Rsun],
        y=[point_source[1]/Rsun, closest_point_on_disk[1]/Rsun],
        z=[point_source[2]/Rsun, closest_point_on_disk[2]/Rsun],
        mode='lines',
        line=dict(color='cyan', width=3, dash='dash'),
        name='Line to Closest Disk Point',
        showlegend=False  # Hide this trace from the legend
    ))

    # Plot the line connecting the point source to the center
    fig.add_trace(go.Scatter3d(
        x=[point_source[0]/Rsun, 0],
        y=[point_source[1]/Rsun, 0],
        z=[point_source[2]/Rsun, 0],
        mode='lines',
        line=dict(color='cyan', width=3, dash='dash'),
        name='Line to Center',
        showlegend=False  # Hide this trace from the legend
    ))

    fig.update_coloraxes(colorbar_title_side='right')
    # Update layout to ensure color bars and plot are well-placed
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Rsun)',
            yaxis_title='Y (Rsun)',
            zaxis_title='Z (Rsun)',
            aspectmode='cube'
        ),
        # title=f'3D Star and Inclined Disk with Temperature Change (Incl. y:{inclination_y_rad}°) & (Inc. z:{inclination_z_rad}°)',
        title=f'',
        autosize=True,
        margin=dict(l=0, r=300, b=0, t=50),  # Adjust margins to fit color bars
        width=1000,  # Fixed width for the entire plot
        height=800,  # Fixed height for the entire plot
        legend=dict(
            x=1.05,  # Position legend outside the plot area
            y=0.5,
            xanchor='left',
            yanchor='middle'
        )
    )
    
    # )
    
    fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=2.5, z=0.3),  # Adjust these values to change the view angle
            center=dict(x=0, y=0, z=0),    # Center of the plot
            up=dict(x=0, y=0, z=1)         # Defines which direction is up
        )
        )
    )
    
        # Update layout to maintain aspect ratio and equal range
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X (Rsun)',
                range=[-1.1*point_source_distance/Rsun, 1.1*point_source_distance/Rsun]
            ),
            yaxis=dict(
                title='Y (Rsun)',
                range=[-1.1*point_source_distance/Rsun, 1.1*point_source_distance/Rsun]
            ),
            zaxis=dict(
                title='Z (Rsun)',
                range=[-1.1*point_source_distance/Rsun, 1.1*point_source_distance/Rsun]
            ),
            aspectmode='auto',  # Allows manual control of axis ranges
        )
    )
    

    fig.show()
    
    
# Define the Planck function for frequencies and wavelengths
def planck_frequency(nu, T):
    return (8 * np.pi * h * nu**3 / c**3) / (np.exp(h * nu / (k * T)) - 1)

def planck_wavelength(lam, T):
    return (2 * h * c**2) / (lam**5 * (np.exp(h * c / (lam * k * T)) - 1))

def B_l(wav,T0):
    return 2*h*c**2/wav**5/(np.exp(h*c/(wav*kb*T0))-1)

def compute_sed_from_disk(x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk,inner_disk_radius, outer_disk_radius, num_points, en='W'):

    r_disk = np.linspace(inner_disk_radius, outer_disk_radius, num_points)
    theta_disk = np.linspace(0, 2 * np.pi, num_points)
    dr = r_disk[1]-r_disk[0]
    dtheta = theta_disk[1]-theta_disk[0]

    # Compute area of each grid cell in the disk
    # Convert disk grid to polar coordinates
    r_disk = np.sqrt(x_disk_inclined**2 + y_disk_inclined**2+z_disk_inclined**2)

    # Area element in polar coordinates
    area_grid_point = r_disk * dr * dtheta

    # Constants
    #T = 6000  # Temperature in Kelvin (example: 6000K, typical for a star's surface)
    wavelength_min = 1e-7  # Minimum wavelength in  (1 nm)
    wavelength_max = 1e-1  # Maximum wavelength

    frequency_min = c / wavelength_max  # Minimum frequency (corresponding to maximum wavelength)
    frequency_max = c / wavelength_min  # Maximum frequency (corresponding to minimum wavelength)

    num_points_en = 1000  # Number of points in the wavelength and frequency arrays

    # Create wavelength and frequency arrays
    wavelengths = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), num_points_en)
    frequencies = np.logspace(np.log10(frequency_min), np.log10(frequency_max), num_points_en)


    # Initialize array for combined SED
    sed = np.zeros_like(frequencies)
    
        # Compute the luminosity and SED
    if en == 'W':
        # Compute Planck's law for each wavelength
        sed = np.zeros_like(wavelengths)
        for i in range(T_new_disk.shape[0]):
            for j in range(T_new_disk.shape[1]):
                T = T_new_disk[i, j]
                area = area_grid_point[i, j]
                luminosity_per_wavelength = planck_wavelength(wavelengths, T) * area*np.pi
                sed += luminosity_per_wavelength
                
    elif en == 'F':
        # Compute Planck's law for each frequency
        sed = np.zeros_like(frequencies)
        for i in range(T_new_disk.shape[0]):
            for j in range(T_new_disk.shape[1]):
                T = T_new_disk[i, j]
                area = area_grid_point[i, j]
                luminosity_per_frequency = planck_frequency(frequencies, T) * area*np.pi
                sed += luminosity_per_frequency

    return wavelengths if en == 'W' else frequencies, sed

#     # Compute the luminosity and SED
#     for i in range(T_new_disk.shape[0]):
#         for j in range(T_new_disk.shape[1]):
#             T = T_new_disk[i, j]
#             area = area_grid_point[i, j]
#             # Compute Planck's law for each frequency
#             luminosity_per_frequency = planck_frequency(frequencies, T) * area
#             # print(T)
#             # Sum over all grid points
#             sed += luminosity_per_frequency
            
#     return frequencies, sed





def sed_build(star_radius, num_points, point_source_distance, T0_star, L, inner_disk_radius, outer_disk_radius, inclination_angle_y, inclination_angle_z, T0_disk, albedo=0.5, n=2, en='W',Carciofi=True):
    
    x_star, y_star, z_star, T_new_star, point_source, x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk = compute_temperature_change_cgs_disk(
        star_radius, num_points, point_source_distance, T0_star, L, inner_disk_radius, outer_disk_radius, inclination_angle_y, inclination_angle_z, T0_disk, albedo=albedo, n=n,Carciofi=Carciofi
    )
    
    frequencies, sed_disk = compute_sed_from_disk(x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk,inner_disk_radius, outer_disk_radius, num_points, en=en)
    
    x_star, y_star, z_star, T_new_star, point_source, x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk = compute_temperature_change_cgs_disk(
        star_radius, num_points, point_source_distance, T0_star, 0.0, inner_disk_radius, outer_disk_radius, inclination_angle_y, inclination_angle_z, T0_disk, albedo=albedo, n=n, Carciofi=Carciofi
    )
    
    frequencies0, sed_disk0 = compute_sed_from_disk(x_disk_inclined, y_disk_inclined, z_disk_inclined, T_new_disk,inner_disk_radius, outer_disk_radius, num_points, en=en)
    
    if en == 'F':
        sed_freq_Star0 = planck_frequency(frequencies, T0_star) *star_radius**2 *4*(np.pi)**2
        
    elif en == 'W':
        sed_freq_Star0 = planck_wavelength(frequencies, T0_star) *star_radius**2 *4*(np.pi)**2
    
    return (frequencies, sed_freq_Star0, sed_disk0, sed_disk)
    
    
    
def col_mag(wave, sed,distance=50,type='S'):
    ogle_V = np.loadtxt('Filters/OGLE-IV-V-filter.dat',usecols=range(0,2))
    ogle_I = np.loadtxt('Filters/OGLE-IV-I-filter.dat',usecols=range(0,2))
    ogle_V = pd.DataFrame(ogle_V,columns=['Wavelength','Mean transmission'])
    ogle_I = pd.DataFrame(ogle_I,columns=['Wavelength','Mean transmission'])

    # wavelengths where the filters work
    wav_V = np.linspace(min(ogle_V['Wavelength']), max(ogle_V['Wavelength']), len(ogle_V))*A_to_cm
    wav_I = np.linspace(min(ogle_I['Wavelength']), max(ogle_I['Wavelength']), len(ogle_I))*A_to_cm
   

    # QE for CCDs
    QE_DAT = np.loadtxt('Filters/OGLE-IV-QE-curve.dat',usecols=range(0,2))
    QE_DF = pd.DataFrame(QE_DAT,columns=['Angstroms','QE'])
    QE = np.interp(x=wav_V, xp=QE_DF['Angstroms']*A_to_cm, fp=QE_DF['QE'])

    #for normalization
    int_trans_V = integrate.trapezoid(QE*ogle_V['Mean transmission']*wav_V, wav_V)
    int_trans_I = integrate.trapezoid(QE*ogle_I['Mean transmission']*wav_I, wav_I)
    # norms = 4*np.pi**2*R_star**2  #solid angle for edge-on disk
    # normD = 2*np.pi*distance**2  #for flux-luminosity conversion
    
    cs = CubicSpline(wave, sed)
    
    sed_int_V = cs(wav_V)
    sed_int_I = cs(wav_I)
    
    # energy flux normalized to the transmission*wavelength integral
    # in units of erg/s/cm/cm^2 or erg/s/Angstrom/cm^2
    f_V = integrate.trapezoid(QE*ogle_V['Mean transmission']*sed_int_V*wav_V, wav_V)/int_trans_V
    f_I = integrate.trapezoid(QE*ogle_I['Mean transmission']*sed_int_I*wav_I, wav_I)/int_trans_I


    # Magnitudes  in Johnson-Cousins system (found in table A2 of https://ui.adsabs.harvard.edu/abs/1998A%26A...333..231B/abstract)
    # taking into consideration the extinction A = E(V-I)*const (found in GAVO 'Reddening and Extinction maps of the Magellanic Clouds')
    mag_V_sys = -2.5*np.log10(f_V*A_to_cm)-21.100 + 0.1128
    mag_I_sys = -2.5*np.log10(f_I*A_to_cm)-21.100-1.271 + 0.06627
    
    # color taking into consideration the extinction E(V-I)=0.047 mag for SMC
    color_VI_sys =  mag_V_sys - mag_I_sys + 0.047 
    
    return mag_V_sys,mag_I_sys,color_VI_sys
    