import os
import numpy as np
from addict import Dict

cfg = Dict()

cfg.name = 'cent_narrow_prior'

cfg.bnn_omega = dict(
                 lens_mass = dict(
                                 profile='PEMD',
                                 center_x = dict(
                                          dist='normal',
                                          mu={'init':0.0,'sigma':0.2,
                                              'prior':uniform(loc=-5,
                                                scale=10).logpdf},
                                          sigma={'init':0.05,'sigma':0.03,
                                              'prior':uniform(loc=0,
                                                scale=10).logpdf}
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu={'init':0.0,'sigma':0.2,
                                              'prior':uniform(loc=-5,
                                                scale=10).logpdf},
                                          sigma={'init':0.05,'sigma':0.03,
                                              'prior':uniform(loc=0,
                                                scale=10).logpdf}
                                          ),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='lognormal',
                                              mu={'init':0.7,'sigma':0.3,
                                                'prior':uniform(loc=-5,
                                                scale=10).logpdf},
                                              sigma={'init':0.02,'sigma':0.01,
                                              'prior':uniform(loc=0,
                                                scale=10).logpdf}
                                              ),
                                 theta_E = dict(
                                                dist='lognormal',
                                                mu={'init':0.0,'sigma':0.3,
                                                  'prior':uniform(loc=-5,
                                                    scale=10).logpdf},
                                                sigma={'init':0.02,'sigma':0.01,
                                                  'prior':uniform(loc=0,
                                                    scale=10).logpdf}
                                                ),
                                 # Beta(a, b)
                                 e1 = dict(
                                           dist='normal',
                                           mu={'init':0.0,'sigma':0.3,
                                            'prior':uniform(loc=-1,
                                                scale=2).logpdf},
                                           sigma={'init':0.03,'sigma':0.01,
                                                  'prior':uniform(loc=0,
                                                    scale=10).logpdf}
                                           ),
                                 e2 = dict(
                                           dist='normal',
                                           mu={'init':0.0,'sigma':0.3,
                                            'prior':uniform(loc=-1,
                                                scale=2).logpdf},
                                           sigma={'init':0.03,'sigma':0.01,
                                                  'prior':uniform(loc=0,
                                                    scale=10).logpdf}
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu={'init':-2.73,'sigma':0.5,
                                                            'prior':uniform(loc=-5,
                                                              scale=10).logpdf},
                                                         sigma={'init':0.1,'sigma':0.05,
                                                            'prior':uniform(loc=0,
                                                              scale=10).logpdf}
                                                         ),
                                       psi_ext = dict(
                                                     dist='uniform',
                                                     upper={'init':0.5*np.pi,'sigma':0.0,
                                                            'prior':uniform(loc=-5,scale=10).logpdf},
                                                     lower={'init':0.5*np.pi,'sigma':0.0,
                                                            'prior':uniform(loc=-5,scale=10).logpdf},
                                                     ),
                                       ),

                 )

