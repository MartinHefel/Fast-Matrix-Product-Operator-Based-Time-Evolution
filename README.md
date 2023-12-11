This repository contains the code developed during the course of my bachelor's thesis:
# Fast Matrix Product Operator Based Time Evolution

### Thesis Information
We propose and benchmark a matrix product operator (MPO) based time evolution which uses a variational QR decomposition based truncation scheme instead of a singular value decomposition (SVD). This improves the scaling of the time evolution algorithm with respect to the local Hilbert space dimension $d$ from $d^3$ to $d^2$. Additionally, we demonstrate that the proposed algorithm runs efficiently on GPUs in contrast to the established SVD based truncation method. This results in an additional, hardware-dependent speedup.

### Code Information
The developed object-based Python library can perform a MPO based time evolution of a MPS with three different truncation methods (SVD, QR and QR+CBE) on two different backends ([NumPy](https://numpy.org/) and [Pytorch](https://pytorch.org/get-started/locally/)). `TeNW_example.py` is an example script that shows how to perform an exemplary time evolution.

The used assumptions are the following: The system is one-dimensional and finite. There are no periodic boundary conditions between the last and the first site. The local Hilbert space dimension $d$ as well as the MPO dimension $D$ are identical for each site. 

In general, models that compile with the assumptions above can be implemented by defining the MPO in the MPO class and referencing it in the respective model class. The implemented models in `TeNW/model.py` are the quantum clock model for nearest-neighbor interactions (`ClockModel_NN`) and nearest- and next-nearest-neighbor interactions (`ClockModel_NNN`). The file and object structure of the library can be found below.

```
TeNW
├── backend.py
│   ├── numpy_backend
│   └── torch_backend
├── mps_network.py
│   └── MPS
├── mpo_network.py
│   ├── MPO_template
│   ├── MPO_TE_ClockModel_NN
│   └── MPO_TE_ClockModel_NNN
├── model.py
│   ├── model_template
│   ├── ClockModel_NN
│   └── ClockModel_NNN
├── truncator.py
│   ├── svd
│   ├── qr
│   └── qr_cbe
└── algorithm.py
    └── time_evolution_engine
```
