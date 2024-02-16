from astropy.io import fits, ascii
from astropy.time import Time
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

def fetch_info_table(ra, dec):
    return tbe = ascii.read(f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci?POS={ra},{dec}&SIZE=0.01")

def fetch_irsa_cut(ra, dec, yr, fd, frd, padf, fltr, cid, qid, size=20):
    """Query IRSA and download the fits file that contais the (ra, dec source). 
       This table requrires sci-metadata info (i.e tbe = ascii.read("https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci?POS=118.77578,56.29234&SIZE=0.01"))
    
       Parameters
         ----------
            ra: float
                Right ascension of the source
            dec: float
                Declination of the source
            yr: int
                Year of the observation
            fd: int
                Field of the observation
            frd: str
                Frame of the observation
            padf: int
                Padf of the observation
            fltr: str
                Filter of the observation
            cid: int
                CCD of the observation
            qid: int    
                Quarter of the observation
            size: int   
                Size of the cutout in arcsec. Default is 20 arcsec.

        Returns
        -------
            fits file: .fits
                Fits file of the cutout
            chi-square psf_info: float
                PSF info of the cutout      
    """
    yr_int = int(yr)
    fd_int = int(fd)
    padf_int = int(padf)
    cid_int = int(cid)

    url = f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{yr_int}/{fd_int:04d}/{frd}/ztf_{yr_int}{fd_int:04d}{frd}_{padf_int:06d}_{fltr}_c{cid_int:02d}_o_q{qid}_sciimg.fits?center={ra},{dec}&size={size}arcsec&gzip=false"
    print (url)
    psf_info = fits.open(f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{yr_int}/{fd_int:04d}/{frd}/ztf_{yr_int}{fd_int:04d}{frd}_{padf_int:06d}_{fltr}_c{cid_int:02d}_o_q{qid}_psfcat.fits?center={ra},{dec}&size={size}arcsec&gzip=false")
    
    sky_src = SkyCoord(psf_info[1].data['ra']*u.deg, psf_info[1].data['dec']*u.deg)
    src = SkyCoord(float(ra)*u.deg, float(dec)*u.deg)
    
    separations = sky_src.separation(src).arcsec
    
    return fits.open(url), psf_info[1].data['chi'][np.argmin(separations)]