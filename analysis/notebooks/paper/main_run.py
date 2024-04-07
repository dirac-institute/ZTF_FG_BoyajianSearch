import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from matplotlib import rcParams
import lsdb
from tqdm import tqdm
import dask
from dask.distributed import Client
dask.config.set({"temporary-directory" :'/epyc/ssd/users/atzanida/tmp'})
dask.config.set({"dataframe.shuffle-compression": 'Snappy'})

import sys
sys.path.insert(1, '../../dipper')
import tools as my_tools
import dipper as dip_pipeline
import models as dip_models
from evaluate import evaluate as evaluate
from evaluate import evaluate_updated
from gpmcmc import model_gp
from evaluate import half_eval as half_eval
import dask.dataframe as dd
from tape import Ensemble, ColumnMapper
pd.options.mode.chained_assignment = None

from distributed import Client
client = Client(n_workers=45, threads_per_worker=1, memory_limit='20GB')
print ("Client: ", client)

# Initialize an Ensemble
ens = Ensemble(client=client)

def main(client=client, ens=ens):
    gaia = lsdb.read_hipscat("/data3/epyc/data3/hipscat/catalogs/gaia_dr3/gaia", 
                        columns=['ra', 'dec', 'parallax', 'parallax_over_error', 
                                'bp_rp', 'solution_id', 
                                'source_id', 
                                'pmra', 'pmra_error', 
                                'pmdec', 'pmdec_error', 
                                'parallax_error', 
                                'phot_g_mean_mag', 
                                'l', 'b', 'non_single_star', 
                                'classprob_dsc_combmod_galaxy', 
                                'classprob_dsc_combmod_star', 
                                 'in_qso_candidates'])

    # load ZTF object table
    ztf = lsdb.read_hipscat("/epyc/data3/hipscat/catalogs/ztf_axs/ztf_dr14")

    # Load ZTF DR14 sources
    ztf_sources = lsdb.read_hipscat("/epyc/data3/hipscat/catalogs/ztf_axs/ztf_zource")

    fgk_object = lsdb.read_hipscat("/nvme/users/atzanida/tmp/sample_final_starhorse_hips")

    _sources = fgk_object.join(
        ztf_sources, left_on="ps1_objid_ztf_dr14", right_on="ps1_objid")
    
    # ColumnMapper Establishes which table columns map to timeseries quantities
    colmap = ColumnMapper(
            id_col='_hipscat_index',
            time_col='mjd',
            flux_col='mag',
            err_col='magerr',
            band_col='band',
        )

    ens.from_dask_dataframe(
        source_frame=_sources._ddf,
        object_frame=fgk_object._ddf,
        column_mapper=colmap,
        sync_tables=False, # Avoid doing an initial sync
        sorted=True, # If the input data is already sorted by the chosen index
        sort=False,
    )

    # feature evaluation 
    column_names = ['Nphot',
        'biweight_scale',
        'frac_above_2_sigma', # in deviation
        'Ndips',
        'rate',
        'chi2dof',
        'skew', 
        'kurtosis',
        'mad',
        'stetson_i',
        'stetson_j',
        'stetson_k',
        'invNeumann',    
        'best_dip_power',
        'best_dip_time_loc',
        'best_dip_start',
        'best_dip_end',
        'best_dip_dt',
        'best_dip_ndet',
        'lc_score']

    # Define DataFrame with loc and scale as meta
    my_meta = pd.DataFrame(columns=column_names, dtype=float)

    calc_ = ens.batch(evaluate_updated,
    'mjd_ztf_zource', 'mag_ztf_zource', 
    'magerr_ztf_zource', 'catflags_ztf_zource',
    'band_ztf_zource',
    meta=my_meta,
    use_map=True)

    ens.object.join(calc_).update_ensemble()

    # DEMO for now... 
    demo = ens.object.head(1_000, npartitions=100)

    # Store...
    demo.to_parquet("/epyc/ssd/users/atzanida/tmp/starH24/DEMO_Source_Computed_April2024.parquet", engine='pyarrow')
