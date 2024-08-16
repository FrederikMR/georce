# GEORCE
Efficient computations of geodesic algorithms formulating optimization of the energy functional as a control problem.

![Constructed geodesics using GEORCE and similar optimization algorithms](https://github.com/user-attachments/assets/b4264569-6fd1-4af3-918b-dad8cfe28b47)



## Installation and Requirements

The implementations in the GitHub is Python using JAX. To clone the GitHub reporsitory and install packages type the following in the terminal

```
git clone https://github.com/FrederikMR/georce.git
cd georce
pip install -r requirements.txt
```

The first line clones the repository, the second line moves you to the location of the files, while the last line install the packages used in repository.

## Code Structure

The following shows the structure of the code. All general implementations of geometry and optimization algorithms can be found in the "geometry" folder for both the Riemannian and Finsler case.

    .
    ├── load_manifold.py                   # Load manifolds and points for connecting geodesic
    ├── runtime.py                         # Times length and runtime for different optimization algorithms to consturct geodesic
    ├── finsler_geodesic.ipynb             # Finsler geometry figures and plots
    ├── riemannian_geodesics.ipynb         # Riemannian geometry figures and plots
    ├── runtime_estimates.ipynb            # Runtime tables and figures
    ├── timing                             # Contains all timing results
    ├── geometry                           # Contains implementation of Finsler and Riemannian geometries as well as geodesic optimization algorithms, inlcuding GEORCE
    ├── LICENSE
    └── README.md

## Reproducing Experiments

All experiments can be re-produced by running the notebooks and the runtime.py package for the given manifold, hyper-parameters and optimization method.

## Logging

All experimental results for the runtime and length estimates are saved as .pkl files in the folder "timing".

## Reference


