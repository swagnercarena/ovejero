import os
import numpy as np
from addict import Dict

cfg = Dict()

cfg.bnn_omega = dict(
                 lens_mass = dict(
                                 profile='PEMD',
                                 center_x = dict(
                                          dist='normal',
                                          mu={'init':0.102,'sigma':0.2,
                                              'upper':np.inf,'lower':-np.inf},
                                          sigma={'init':0.05,'sigma':0.03,
                                              'upper':np.inf,'lower':0}
                                          ),
                                 center_y = dict(
                                          dist='normal',
                                          mu={'init':-0.102,'sigma':0.2,
                                              'upper':np.inf,'lower':-np.inf},
                                          sigma={'init':0.05,'sigma':0.03,
                                              'upper':np.inf,'lower':0}
                                          ),
                                 # Lognormal(mu, sigma^2)
                                 gamma = dict(
                                              dist='lognormal',
                                              mu={'init':0.8,'sigma':0.3,
                                                'upper':np.inf,'lower':-np.inf},
                                              sigma={'init':0.02,'sigma':0.01,
                                              'upper':np.inf,'lower':0}
                                              ),
                                 theta_E = dict(
                                                dist='lognormal',
                                                mu={'init':-0.1,'sigma':0.3,
                                                  'upper':np.inf,'lower':-np.inf},
                                                sigma={'init':0.02,'sigma':0.01,
                                                  'upper':np.inf,'lower':0}
                                                ),
                                 # Beta(a, b)
                                 e1 = dict(
                                           dist='normal',
                                           mu={'init':-0.2,'sigma':0.3,
                                            'upper':np.inf,'lower':-np.inf},
                                           sigma={'init':0.03,'sigma':0.01,
                                                  'upper':np.inf,'lower':0}
                                           ),
                                 e2 = dict(
                                           dist='normal',
                                           mu={'init':-0.2,'sigma':0.3,
                                            'upper':np.inf,'lower':-np.inf},
                                           sigma={'init':0.03,'sigma':0.01,
                                                  'upper':np.inf,'lower':0}
                                           ),
                                 ),

                 external_shear = dict(
                                       profile='SHEAR_GAMMA_PSI',
                                       gamma_ext = dict(
                                                         dist='lognormal',
                                                         mu={'init':-1.3,'sigma':0.5,
                                                            'upper':np.inf,'lower':-np.inf},
                                                         sigma={'init':0.1,'sigma':0.05,
                                                            'upper':np.inf,'lower':0}
                                                         ),
                                       psi_ext = dict(
                                                     dist='uniform',
                                                     upper={'init':0.5*np.pi,'sigma':0.0,
                                                            'upper':0.5*np.pi,'lower':0.5*np.pi},
                                                     lower={'init':0.5*np.pi,'sigma':0.0,
                                                            'upper':-0.5*np.pi,'lower':-0.5*np.pi},
                                                     ),
                                       ),

                 )

