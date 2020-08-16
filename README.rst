==========================================================================
ovejero - Bayesian Neural Network Inference of Strong Gravitational Lenses
==========================================================================

.. image:: https://travis-ci.com/swagnercarena/ovejero.png?branch=master
	:target: https://travis-ci.org/swagnercarena/ovejero

.. image:: https://readthedocs.org/projects/ovejero/badge/?version=latest
	:target: https://ovejero.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://coveralls.io/repos/github/swagnercarena/ovejero/badge.svg?branch=master
	:target: https://coveralls.io/github/swagnercarena/ovejero?branch=master

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/swagnercarena/ovejero/LICENSE

''ovejero'' conducts hierarchical inference of strongly-lensed systems with Bayesian neural networks

Installation
------------

Lenstronomy requires an additional fortran package (fastell) to run lens models with elliptical mass distributions. Thankfully, installing the package is fairly easy (although a fortran compiler is required).

.. code-block:: bash

    $ git clone https://github.com/sibirrer/fastell4py.git <desired location>
    $ cd <desired location>
    $ python setup.py install --user


In the future, ovejero will be a pypi package. For now, it can be installed by cloning the git repo.

.. code-block:: bash

	$ git clone https://github.com/swagnercarena/ovejero.git
	$ cd overjero/
	$ pip install -e . -r requirements.txt

The addition of the -e option will allow you to pull ovejero updates and have them work automatically.

Main Features
-------------

* Ability to train Bayesian Neural Networks (BNNs) to predict posteriors on strong gravitational lensing images
* Integration with forward modeling tools in lenstronomy to allow comparison between BNN outputs and more traditional methods
* Hierarchical Inference tools that allow for population parameter estimates and unbiased posteriors on independent test sets

Demos
-----

The jupyter notebooks in the demos folder walk you through how to use all the main features of the ovejero package.

* `Generating a Configuration File for Training a Model <https://github.com/swagnercarena/ovejero/blob/master/demos/Generate_Config.ipynb>`_: Learn how to use json function in python to write out a configuration file for model training.
* `Fitting a Model Using model_trainer <https://github.com/swagnercarena/ovejero/blob/master/demos/Train_Toy_Model.ipynb>`_: Learn how to use model_trainer to fit the types of models used by ovejero.
* `Testing the Performance of a Model That Has Been Fit <https://github.com/swagnercarena/ovejero/blob/master/demos/Test_Model_Performance.ipynb>`_: Learn how to test the performance of a trained model on the validation set.
* `Comparing Forward Modeling to a BNN Posterior <https://github.com/swagnercarena/ovejero/blob/master/demos/Forward_Modeling_Demo.ipynb>`_: Learn how to compare the performance of the BNN model to a forward modeling approach.
* `Hierarchical Inference on a Test Set <https://github.com/swagnercarena/ovejero/blob/master/demos/Hierarchical_Inference_Demo.ipynb>`_: Learn how to run hierarchical inference on a test set using a trained BNN.


