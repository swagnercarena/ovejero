import os
import numpy as np
from addict import Dict
from scipy.stats import uniform

cfg = Dict()

cfg.bnn_omega = dict(
                 lens_mass = dict(
                                 profile='PEMD',
                                 center_x = dict(
                                          dist='normal',
                                          mu={'init':0.0,'sigma':0.2,
                                              'prior':uniform(loc=-5,
                                                scale=10).logpdf},
                                          sigma={'init':0.102,'sigma':0.03,
                                              'prior':uniform(loc=0,
                                                scale=10).logpdf}
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu={'init':0.0,'sigma':0.2,
                                              'prior':uniform(loc=-5,
                                                scale=10).logpdf},
                                          sigma={'init':0.102,'sigma':0.03,
                                              'prior':uniform(loc=0,
                                                scale=10).logpdf}
                                          ),
                                 phi = dict(
                                            dist='uniform',
                                            upper={'init':0.5*np.pi,'sigma':0.0,
                                                  'prior':uniform(loc=-5*np.pi,
                                                    scale=10*np.pi).logpdf},
                                            lower={'init':-0.5*np.pi,'sigma':0.0,
                                                  'prior':uniform(loc=-5*np.pi,
                                                    scale=10*np.pi).logpdf},
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu={'init':-2.73,'sigma':0.5,
                                                            'prior':uniform(loc=-5,
                                                              scale=10).logpdf},
                                                         sigma={'init':1.05,'sigma':0.05,
                                                            'prior':uniform(loc=0,
                                                              scale=10).logpdf}
                                                         ),
                                       psi_ext = dict(
                                                     dist='uniform',
                                                     upper={'init':0.5*np.pi,'sigma':0.0,
                                                            'prior':uniform(loc=-5*np.pi,
                                                              scale=10*np.pi).logpdf},
                                                     lower={'init':-0.5*np.pi,'sigma':0.0,
                                                            'prior':uniform(loc=-5*np.pi,
                                                              scale=10*np.pi).logpdf},
                                                     ),
                                       ),

                 cov_info = dict(
                                # List of 2-tuples specifying which params are correlated
                                cov_params_list=[
                                ('lens_mass_theta_E'),
                                ('lens_mass_q'),
                                ('lens_mass_gamma')
                                ],
                                cov_omega = dict(
                                                # Whether each param is log-parameterized
                                                is_log=[
                                                True,
                                                True,
                                                True,
                                                ],
                                                # The mean vector
                                                mu={'init':np.array([0.242, -0.408, 0.696]),
                                                    'sigma':np.array([0.1,0.1,0.1]),
                                                    'prior':[uniform(loc=-5,
                                                      scale=10).logpdf]*3
                                                },
                                                tril={'init':np.array([0.5,0.5,0.5,0.4,0.4,0.4]),
                                                      'sigma':np.array([0.5,0.5,0.5,0.4,0.4,0.4]),
                                                      'prior':[uniform(loc=0,scale=10).logpdf,
                                                        uniform(loc=-5,scale=10).logpdf,
                                                        uniform(loc=0,scale=10).logpdf,
                                                        uniform(loc=-5,scale=10).logpdf,
                                                        uniform(loc=-5,scale=10).logpdf,
                                                        uniform(loc=0,scale=10).logpdf]},
                                                 ),
                                )
                 )

