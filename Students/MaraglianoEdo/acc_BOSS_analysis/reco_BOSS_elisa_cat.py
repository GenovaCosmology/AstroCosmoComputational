


# This script reconstructs the redshift-space galaxy distribution of the EuclidLargeMocks
# using the Zel'dovich Approximation (ZA) method. The reconstructed catalogues are saved
# in the local directory specified by the user. 
# Upload to the Euclid repository is not yet implemented.
# If submit_2pcf_measurement = True, the script also submits the measurement of the 2-point correlation
# function of the reconstructed catalogues

# Usage: python reco_EuclidLargeMocks.py emisphere zname rec 


#---------- Import modules ---------
import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import Recon_challenge.GetData as GetData
import Recon_challenge.Reconstruction as Rec
from Recon_challenge.Covariance import PkCamb as PkCamb
import logging
from MyThesisLibrary_dev.farm_submitters import two_pcf_submitter as sub
import matplotlib.pyplot as plt

import pymangle


#---------- Set parameters ---------

emisphere = str(sys.argv[1])                #emisphere
zname = str(sys.argv[2])                    #redshift id
rec = str(sys.argv[3])                      #reconstruction type

perform_ZA_reconstruction = True
submit_2pcf_measurement = True

z_rotation = False



# ---------- fixed parameters ----------

zdict = {'z1': 0,'z2':1}
z_id = zdict[zname]

#Data
DR = "BOSSDR12" #Data release
space = 'RedshiftSpace'

#reconstruction
los_ax = None                                  # line-of-sight must be None on the lightcone, so that the code automatically sets it to the correct axis
nthreads = 48                                  # number of treads for parallel computing
nmesh = 512                                    # mesh for density interpolation (should be : > mean interparticle) distance
          

# ----------------- logging -----------------
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logfile_path = f'/home/emaragliano/Work/Projects/acc_BOSS_analysis/ElisaBOSS_reconstructed_{emisphere}_{zname}.log'
file_handler = logging.FileHandler(logfile_path)        
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


logger.info(f'zname = {zname}') 
logger.info(f'nmesh = {nmesh}')    

# ----------------- modified version of GetAngPos -----------------

def GetAngPositions(in_file, weight_type='WEIGHT_TOT'):
    """
    Get angular positions from a FITS file.

    Parameters:
    in_file (str): The path to the input FITS file.

    Returns:
    tuple: A tuple containing the angular positions and weights.
        - ang_pos (ndarray): A 2D array of shape (3, Ng) containing the RA, DEC, and REDSHIFT values.
        - weights (ndarray): A 1D array of shape (Ng) containing the weights.

    """
    hdul = fits.open(in_file)
    hdr=hdul[1].header
    data=hdul[1].data

    Ng=len(data['RA'])
    ang_pos=np.empty((3,Ng), dtype=np.float64)
    if not (weight_type in data.columns.names):
        weights=np.ones(Ng, dtype=np.float64)
        print('No weights found, setting all weights to 1')
    else:
        weights=np.array(data[weight_type])
        print('using weights from column ', weight_type)

    ang_pos[0,:]=np.array(data['RA'])
    ang_pos[1,:]=np.array(data['DEC'])
    ang_pos[2,:]=np.array(data['REDSHIFT'])

    return ang_pos,weights
  

# ----------------- input and output directories -----------------

# local parent directory
local_parent_dir = '/home/emaragliano/Work/Projects/myfarm-disk/BossAnalysis/Elisa_BOSS_catalogues/'
if(os.path.exists(local_parent_dir)==False):
    raise FileNotFoundError('Local parent directory does not exist')


# path to data catalogues
paths_to_data = {
    'z1_North': os.path.join(local_parent_dir,f'data','galaxy_North_z1.fits' ),
    'z2_North': os.path.join(local_parent_dir,f'data','galaxy_North_z2.fits'),
    'z1_South': os.path.join(local_parent_dir,f'data','galaxy_South_z1.fits'),
    'z2_South': os.path.join(local_parent_dir,f'data','galaxy_South_z2.fits')
}

# path to mangle mask
path_to_mask = {
    'North': os.path.join(local_parent_dir,'masks','mask_DR12v5_CMASSLOWZTOT_North.ply' ),
    'South': os.path.join(local_parent_dir,'masks','mask_DR12v5_CMASSLOWZTOT_South.ply'),
}

# path to random catalogues
paths_to_random = {
    'z1_North': os.path.join(local_parent_dir,f'random','random0_North_z1.fits' ),
    'z2_North': os.path.join(local_parent_dir,f'random','random0_North_z2.fits'),
    'z1_South': os.path.join(local_parent_dir,f'random','random0_South_z1.fits'),
    'z2_South': os.path.join(local_parent_dir,f'random','random0_South_z2.fits')
}

# local directory for saving reconstructed catalogues
local_dir_reconstruction = os.path.join(local_parent_dir, 'reconstructed/')

if(os.path.exists(local_dir_reconstruction)==False):
    os.makedirs(local_dir_reconstruction, exist_ok=True)
    logger.info(f'directory for reconstruction catalogues created at  {local_dir_reconstruction}')

# ----------------- upload options -----------------

#Upload data
Upload = False      #boolean option for uploading catalogues to Euclid repository
                    #use True only during official analysis !!!

ClearLocal = False   #True if you want to remove local catalogues after upload
Root_remote_dir = "data/DR" + str(DR) + "/" + space + "_ZAreconstructed/Smth_test/"

# ----------------- cosmology, smoothing scale, redshift bin -----------------

# bias fitted from the mocks
boss_measured_bias = np.array([1.98,1.8])

#fiducial cosmology of the mocks      
cosmo = GetData.EuclidCosmology()
cosmo['Omega_m'] = 0.320
cosmo['bias'] = boss_measured_bias
cosmo['z'] = np.array([0.38, 0.57])


# smoothing scale
smth = np.array([15])

# ----------------- modified functions for output -----------------

def WriteZAReconstructedPos(pos, weight, emisphere, zname, out_dir):
    
    os.makedirs(out_dir, exist_ok=True)
    hdu = fits.BinTableHDU(Table([pos[0], pos[1], pos[2], weight], names=('RA', 'DEC', 'REDSHIFT', 'WEIGHT_FKP')))
    file_name = f"myBOSS_DR12_CMASSLOWZTOT_"+emisphere+"_"+zname+".fits"
    
    hdu.name = 'CATALOG'
    hdu.header["EXTNAME"] = 'CATALOG '
    hdu.header["FILENAME"] = file_name
    hdu.header["TELESCOP"] = 'BOSS '
    hdu.header["INSTRUME"] = "BOSSDR12"
    hdu.header["CAT_TYPE"] = 'NOT_PROXY'
    hdu.header["CAT_NAME"] = ""
    hdu.header["COORD"] = "EQUATORIAL"
    hdu.header["ANGLE"] = "DEG"
    hdu.header["SELECT"] = "CMASSLOWZTOT_"+emisphere+"_RECONSTRUCTED"

    out_path = os.path.join(out_dir,file_name)
    hdu.writeto(out_path, overwrite=True)
    logger.info(f'\n file saved at: {out_path}')
    
    return True

def WriteZAShiftedRandom(pos, weight, emisphere, zname,out_dir):
    
    os.makedirs(out_dir, exist_ok=True)
    hdu = fits.BinTableHDU(Table([pos[0], pos[1], pos[2], weight], names=('RA', 'DEC', 'REDSHIFT', 'WEIGHT_FKP')))
    file_name = f"myBOSS_DR12_CMASSLOWZTOT_"+emisphere+"_Randoms_"+zname+".fits"
    
    hdu.name = 'CATALOG'
    hdu.header["EXTNAME"] = 'CATALOG '
    hdu.header["FILENAME"] = file_name
    hdu.header["TELESCOP"] = 'BOSS '
    hdu.header["INSTRUME"] = "BOSS_DR12"
    hdu.header["CAT_TYPE"] = 'NOT_PROXY'
    hdu.header["CAT_NAME"] = ""
    hdu.header["COORD"] = "EQUATORIAL"
    hdu.header["ANGLE"] = "DEG"
    hdu.header["SELECT"] = f"CMASSLOWZTOT_{emisphere}_SHIFTED_RANDOMS"

    out_path = os.path.join(out_dir,file_name)
    hdu.writeto(out_path, overwrite=True)
    logger.info(f'\n file saved at: {out_path}')
    
    return True

# ----------------- end of modified functions for output -----------------

# it would be nice to have a function that checks if the catalogues are already there
# if not, it would be nice to have a function that downloads them from the Euclid repository
# with Alfonso code

# ----------------- run reconstruction -----------------
   
logger.info(f'starting code...')
logger.info(f'rec type is {rec}')
logger.info(f'zname is {zname}')
logger.info(f'emisphere is {emisphere}')


# folder containing the pre-rec catalogue
#catalogue_local_dir = os.path.join(local_dir_data, zname)

# path to the data catalogue
#data_path = os.path.join(catalogue_local_dir, f'myBOSS_DR12_CMASSLOWZTOT_{emisphere}_{zname}.fits')
data_path = paths_to_data[zname+'_'+emisphere]
if(os.path.exists(data_path)==False):
    raise FileNotFoundError(f'Data catalogue does not exist at {data_path}')

# random path for the selected snapshot
random_path = paths_to_random[zname+'_'+emisphere]
if(os.path.exists(random_path)==False):
    raise FileNotFoundError(f'Random catalogue does not exist at {random_path}')

#bias and linear growth factor
bias = cosmo['bias'][z_id] 

#set measured bias from reference paper
logger.info(f'measured value for bias at z={zname} is {bias}')

#get fiducial f at the z of the snapshot
k,pkl,f = PkCamb(z_id,cosmo,False)
logger.info(f'fiducial value for f at z={zname} is {f}')

# loop over reconstruction types
#for rec in ZArectype:                        
    
# create output directory for reconstructed catalogues at given mock_id and zname
out_dir = os.path.join(local_dir_reconstruction, rec, zname)
os.makedirs(out_dir, exist_ok=True)

# loop over smoothing scales
for sm in smth:              
    
    out_dir_smoothing = os.path.join(out_dir, f'Smth_{sm}/')
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f'working on catalogue: {DR},{emisphere},{zname}')
    
    if(perform_ZA_reconstruction):

        logger.info(f'\n RUN reconstruction')

        #get positions
        logger.info(f'importing data positions and weights...')
        pos,w = GetAngPositions(data_path, weight_type='WEIGHT_TOT')                      #get data positions

        logger.info(f'DATA positions imported')
        logger.debug(f'type of pos = {type(pos)}')
        logger.info(f'shape of pos = {pos.shape}')
        logger.info(f'shape of w = {w.shape}')
        logger.info(f'w[:10] = {w[:10]}')

        ra_min = np.min(pos[0])
        ra_max = np.max(pos[0])
        dec_min = np.min(pos[1])
        dec_max = np.max(pos[1])
        redshift_min = np.min(pos[2])
        redshift_max = np.max(pos[2])

        logger.info(f'GEOMETRY OF DATA CATALOGUE')
        logger.info(f'pos_min (RA, Dec, redshift) = {ra_min} {dec_min}  {redshift_min}')
        logger.info(f'pos_max (RA, Dec, redshift) = {ra_max} {dec_max}  {redshift_max}')

        # get random positions
        logger.info(f'importing RANDOM positions and weights...')
        posR,wR = GetAngPositions(random_path, weight_type='WEIGHT_FKP')  #get random positions
        logger.info(f'random positions imported')

        # print some info about the geometry of the randoms
        raR_min = np.min(posR[0])
        raR_max = np.max(posR[0])
        decR_min = np.min(posR[1])
        decR_max = np.max(posR[1])
        redshiftR_min = np.min(posR[2])
        redshiftR_max = np.max(posR[2])
        logger.info(f'GEOMETRY OF RANDOM CATALOGUE')
        logger.info(f'posR_min (RA, Dec, redshift) = {raR_min} {decR_min}  {redshiftR_min}')
        logger.info(f'posR_max (RA, Dec, redshift) = {raR_max} {decR_max}  {redshiftR_max}')

        # NO SHIFT IN LIGHTCONE due to line of sight
        
        # compute ZA reconstruction on the lightcone: it internally converts to equatorial coordinates and back
        logger.info('COMPUTING ZA RECONSRUCTION...')
        ang_shft_pos, ang_shft_r = Rec.ZArecon_lightcone(cosmo, ang_pos=pos,w=w,ang_rand=posR,wra=wR,bias=bias,fz=f,los_ax=los_ax,\
                        rectype=rec,nth=nthreads,nmesh=nmesh,smth=sm, logger=logger, z_alignment=z_rotation)
        logger.info('ZA RECONSTRUCTION DONE')
        logger.info(f'shifted data shape = {ang_shft_pos.shape}, shifted random shape = {ang_shft_r.shape}')

        logger.info(f'GEOMETRY OF THE RECONSTRUCTED CATALOGUES - PRE TRIMMING')
        logger.info(f'shft_pos min (RA, Dec, redshift)= {np.min(ang_shft_pos[0])} {np.min(ang_shft_pos[1])}  {np.min(ang_shft_pos[2])}')
        logger.info(f'shft_pos max (RA, Dec, redshift) = {np.max(ang_shft_pos[0])} {np.max(ang_shft_pos[1])}  {np.max(ang_shft_pos[2])}')
        logger.info(f'shft_r min (RA, Dec, redshift) = {np.min(ang_shft_r[0])} {np.min(ang_shft_r[1])}  {np.min(ang_shft_r[2])}')
        logger.info(f'shft_r max (RA, Dec, redshift) = {np.max(ang_shft_r[0])} {np.max(ang_shft_r[1])}  {np.max(ang_shft_r[2])}')
        
        # TRIM BORDERS to avoid data particles outside the lighcone
        logger.info('TRIMMING borders shifted data...')
        logger.debug('reading mangle mask from ', path_to_mask[emisphere])

        mask_path = path_to_mask[emisphere]

        if not(os.path.exists(mask_path)):
            logger.error('Mangle mask not found')
            raise FileNotFoundError('Mangle mask not found')
    
        mangle_mask = pymangle.Mangle(mask_path)
        mask = mangle_mask.contains(ang_shft_pos[0], ang_shft_pos[1])
        logger.debug(f'mask shape = {mask.shape}')
        
        # apply mask to shifted data
        w_s = w[mask]
        ang_shft_pos = ang_shft_pos[:,mask]
        logger.info(f'DATA TRIMMING DONE. shft_pos shape = {ang_shft_pos.shape}, w_s shape = {w_s.shape}')
        logger.info(f'trimmed {-len(pos[0])+len(ang_shft_pos[0])} particles')

        #trim borders to avoid random particles outside the lighcone
        logger.info('TRIMMING borders random...')
        maskR = mangle_mask.contains(ang_shft_r[0], ang_shft_r[1])

        # apply mask to shifted randoms
        wR_s = wR[maskR]
        ang_shft_r = ang_shft_r[:,maskR]

        # print some info about the geometry of the data and randoms after trimming
        logger.info(f'RANDOM TRIMMING DONE. shft_r shape = {ang_shft_r.shape}, wR_s shape = {wR_s.shape}')
        logger.info(f'trimmed {-len(posR[0])+len(ang_shft_r[0])} random particles')

        #save results on local directory
        logger.info('saving catalogues...')
        WriteZAReconstructedPos(ang_shft_pos, w_s, emisphere, zname, out_dir_smoothing)
        WriteZAShiftedRandom(ang_shft_r,wR_s, emisphere, zname, out_dir_smoothing)
        logger.info(f'catalogues saved at {out_dir_smoothing}')

        logger.info(f'\n END reconstruction')
        logger.info(f'--------------------------------------------------')

    else:
        logger.info(f'perform_ZA_reconstruction = {perform_ZA_reconstruction}')
        logger.info(f'no reconstruction performed. Check the boolean variable perform_ZA_reconstruction in the code.')    

    if(perform_ZA_reconstruction):
        ############ -------- plot the trimmed catalogue ---------- ############
        # Get the RA, Dec, and redshift values from ang_shft_pos
        ra = ang_shft_pos[0]
        dec = ang_shft_pos[1]
        redshift = ang_shft_pos[2]

        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the RA vs. Dec projection
        axs[0].scatter(ra, dec, s=1)
        axs[0].set_xlabel('RA')
        axs[0].set_ylabel('Dec')
        axs[0].set_title('RA vs. Dec')

        # Plot the RA vs. redshift projection
        axs[1].scatter(ra, redshift, s=1)
        axs[1].set_xlabel('RA')
        axs[1].set_ylabel('Redshift')
        axs[1].set_title('RA vs. Redshift')

        # Plot the Dec vs. redshift projection
        axs[2].scatter(dec, redshift, s=1)
        axs[2].set_xlabel('Dec')
        axs[2].set_ylabel('Redshift')
        axs[2].set_title('Dec vs. Redshift')

        # Adjust the spacing between subplots
        plt.tight_layout()

        plt.savefig(f'trimmed_catalogue_{emisphere}_{zname}.png')

        ############ -------- end of plot the trimmed catalogue ---------- ############

# ----------------- run 2pcf measurement -----------------
    if(submit_2pcf_measurement):
        logger.info(f'trying to launch the measurement of xi as a new job...')

        sub.submit_auto_2pcf_rec_BOSS_routine_ElisaCat([zname], [emisphere], [rec], smoothing=15, \
            queue='long', threads=40, template_auto_ini_path='pair_count_post_rec_BOSS.ini', logger=logger)