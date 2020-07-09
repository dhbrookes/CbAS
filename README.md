# Conditioning by Adaptive Sampling for Robust Design

This repo contains the code for the paper:

D. H. Brookes, H. Park, and J. Listgarten. Conditioning by adaptive sampling for robust design. *Proceedings of ICML*, 2019.

The most important bits of code are in the files ```src/optimization_algs.py``` and ```notebooks/toy_conditioning.ipynb```. In particular the function ```weighted_ml_opt``` in ```src/optimization_algs.py```, with ```weights_type='cbas'``` runs the central CbAS method. Additionally, ```notebooks/toy_conditioning.ipynb```, is a self-contained iPython notebook that runs the CbAS tests on the toy problem shown in Figure 1.

