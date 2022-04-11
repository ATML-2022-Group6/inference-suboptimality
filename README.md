# inference-suboptimality

Reproduction of "Inference Suboptimality in Variational Autoencoders" [1].

Python Files:
- `vae.py` --- A general purpose VAE class that accepts an encoder/decoder structure and hyperparameter for the approximate posterior distribution to use.
- `train_vae.py` --- Implementation of VAE training algorithm, again parameterised by hyperparameters specified in this file.
- `local_opt.py` --- Implementation of local optimisation using a fully factorized Gaussian or auxiliary flow  approximate distribution. 
- `ais.py` --- Implementation of Annealed Importance Sampling and batching it across the dataset along with computing the IWAE bounds.
- `flow.py` --- Normalising flow with auxiliary variable implementation.
- `hmc.py` --- Implementation of Hamiltonian Monte Carlo for use in AIS.
- `datasets.py` --- Dataset loading code for MNIST and Fashion-MNIST, for both of these the datasets themselves are included in `datasets/`.
- `utils.py` --- Various common functions including loading/saving parameters and some mathematical operations.

Notebooks:
- `run_train_vae.ipynb` --- Notebook for training a VAE model, saving the results and visualising the resulting reconstructions and latent space.
- `run_local_opt.ipynb` --- Runs local optimisation using the trained VAE decoder given specified hyperparameters.
- `run_ais.ipynb` --- Computes the AIS estimates on a trained decoder. 
- `run_visualisation.ipynb` --- Generates the plots for Experiment 5.1

### References

[1] C. Cremer, X. Li, and D. Duvenaud. Inference Suboptimality in Variational Autoencoders. ICML, 2018.