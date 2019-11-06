========================================================
ovejero - Bayesian Neural Network Inference of Strong Gravitational Lenses
========================================================
''ovejero'' conducts hierarchical inference of strongly-lensed systems with Bayesian neural networks

Installation
------------
To run ovejero, you must first install tensorflow 2.0. This is not part of the package requirements for the code because the exact version of tensorflow you will want to install depends on wether or not you have a gpu. If you do not have a gpu, install tensorflow using the command

.. code-block:: bash

	$ pip install tensorflow --user

Otherswise use the command 

.. code-block:: bash

	$ pip install tensorflow-gpu --user

Then clone the directory and run the setup.py file to complete the baobab installation

.. code-block:: bash

	$ python setup.py install

