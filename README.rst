==========================================================================
ovejero - Bayesian Neural Network Inference of Strong Gravitational Lenses
==========================================================================

.. image:: https://travis-ci.com/swagnercarena/ovejero.png?branch=master
	:target: https://travis-ci.org/swagnercarena/ovejero

.. image:: https://readthedocs.org/projects/ovejero/badge/?version=latest
	:target: https://ovejero.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/swagnercarena/ovejero/LICENSE

''ovejero'' conducts hierarchical inference of strongly-lensed systems with Bayesian neural networks

Installation
------------
To run ovejero, you must first install tensorflow 2.0. This is not part of the package requirements for the code because the exact version of tensorflow you will want to install depends on wether or not you have a gpu. If you do not have a gpu, install tensorflow using the command

.. code-block:: bash

	$ pip install tensorflow==2.0.0a0 --user

Otherswise use the command

.. code-block:: bash

	$ pip install tensorflow-gpu==2.0.0b1 --user
	$ pip uninstall gast
	$ pip install gast==0.2.2 --user --no-cache-dir

The gast lines are a workaround for current issues with tensorflow and gast.
Then clone the directory and run the setup.py file to complete the installation

.. code-block:: bash

	$ git clone https://github.com/swagnercarena/ovejero.git
	$ cd overjero/
	$ python setup.py install

