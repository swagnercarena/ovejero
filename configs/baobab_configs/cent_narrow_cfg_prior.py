import os
import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'gamma'
cfg.seed = 1123 # random seed
cfg.bnn_prior_class = 'DiagonalBNNPrior'
cfg.n_data = 1024 # number of images to generate
cfg.train_vs_val = 'val'
cfg.out_dir = os.path.join('out_data/{:s}_{:s}_{:s}_seed{:d}'.format(cfg.name,
                                                        cfg.train_vs_val,
                                                        cfg.bnn_prior_class,
                                                        cfg.seed))
cfg.components = ['lens_mass', 'external_shear', 'src_light',]

cfg.selection = dict(
                 magnification=dict(
                                    min=2.0
                                    ),
                 initial=["lambda x: x['lens_mass']['theta_E'] > 0.5",],
                 )

cfg.instrument = dict(
              pixel_scale=0.08, # scale (in arcseonds) of pixels
              ccd_gain=2.5, # electrons/ADU (analog-to-digital unit). A gain of 8 means that the camera digitizes the CCD signal so that each ADU corresponds to 8 photoelectrons.
              read_noise=4.0, # std of noise generated by read-out (in units of electrons)
              )

cfg.bandpass = dict(
                magnitude_zero_point=25.9463, # (effectively, the throuput) magnitude in which 1 count per second per arcsecond square is registered (in ADUs)
                )

cfg.observation = dict(
                  exposure_time=5400.0, # exposure time per image (in seconds)
                  sky_brightness=22, # sky brightness (in magnitude per square arcseconds)
                  num_exposures=1, # number of exposures that are combined
                  background_noise=None, # overrides exposure_time, sky_brightness, read_noise, num_exposures
                  )

cfg.psf = dict(
           type='PIXEL', # string, type of PSF ('GAUSSIAN' and 'PIXEL' supported)
           kernel_size=91, # dimension of provided PSF kernel, only valid when profile='PIXEL'
           fwhm=0.1, # # full width at half maximum of the PSF (if not specific psf_model is specified)
           which_psf_maps=None, # None if rotate among all available PSF maps, else seed number of the map to generate all images with that map
           )

cfg.numerics = dict(
                supersampling_factor=1)

cfg.image = dict(
             num_pix=64, # cutout pixel size
             inverse=False, # if True, coord sys is ra to the left, if False, to the right
             )

cfg.bnn_omega = dict(
                 lens_mass = dict(
                                 profile='PEMD', # only available type now
                                 # Normal(mu, sigma^2)
                                 center_x = dict(
                                          dist='normal', # one of ['normal', 'beta']
                                          mu=0.0,
                                          sigma=0.05, # two pixels
                                          hyper_prior=dict(
                                                     mu = [-np.inf,np.inf],
                                                     sigma = [0,np.inf])
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu=0.0,
                                          sigma=0.05, # two pixels
                                          hyper_prior=dict(
                                                     mu = [-np.inf,np.inf],
                                                     sigma = [0,np.inf])
                                          ),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='lognormal',
                                              mu=0.7,
                                              sigma=0.01,
                                              hyper_prior=dict(
                                                     mu = [-np.inf,np.inf],
                                                     sigma = [0,np.inf])
                                              ),
                                 theta_E = dict(
                                                dist='lognormal',
                                                mu=0.0,
                                                sigma=0.01,
                                                hyper_prior=dict(
                                                     mu = [-np.inf,np.inf],
                                                     sigma = [0,np.inf])
                                                ),
                                 # Beta(a, b)
                                 e1 = dict(
                                           dist='normal',
                                           mu=0.0,
                                           sigma=0.03,
                                           hyper_prior=dict(
                                                     mu = [-np.inf,np.inf],
                                                     sigma = [0,np.inf])
                                           ),
                                 e2 = dict(
                                           dist='normal',
                                           mu=0.0,
                                           sigma=0.03,
                                           hyper_prior=dict(
                                                     mu = [-np.inf,np.inf],
                                                     sigma = [0,np.inf])
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu=-2.73, # See overleaf doc
                                                         sigma=0.1,
                                                         hyper_prior=dict(
                                                                    mu = [-np.inf,np.inf],
                                                                    sigma = [0,np.inf])
                                                         ),
                                       psi_ext = dict(
                                                     dist='generalized_normal',
                                                     mu=0.0,
                                                     alpha=0.5*np.pi,
                                                     p=10.0,
                                                     hyper_prior = dict(
                                                                 mu = [-np.inf,np.inf],
                                                                 alpha = [0,np.inf],
                                                                 p = [0,np.inf])
                                                     ),
                                       ),

                 )

