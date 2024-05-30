# LINTEL

This repository implements the method proposed in "A Gaussian Process-based Streaming Algorithm for Prediction of Time Series With Regimes and Outliers". This method seeks to improve upon the INTEL algorithm, presented by Liu et al. in "Sequential Online Prediction in the Presence of Outliers and Change Points: an Instant Temporal Structure Learning Approach", which is also implemented.

# Citation
If you use any code or results from this project, please consider citing the LINTEL paper:

```
@inproceedings{waxman2024lintel,
  author = {Waxman, Daniel and Djurić, Petar {M.}},
  booktitle = {2024 27th International Conference on Information Fusion (FUSION)},
  title = {A Gaussian Process-based Streaming Algorithm for Prediction of Time Series With Regimes and Outliers},
  year = {2024},
  note = {Accepted.},
}
```

If you use results related to INTEL in particular, please also consider citing their paper:

```
@article{liu2020sequential,
  title={Sequential Online Prediction in the Presence of Outliers and Change Points: an Instant Temporal Structure Learning Approach},
  author={Liu, Bin and Qi, Yu and Chen, Ke-Jia},
  journal={Neurocomputing},
  volume={413},
  pages={240--258},
  year={2020},
  publisher={Elsevier}
}
```

# Installation Instructions 

To install LINTEL, you can download the git repository and install the package using `pip`:

```
git clone https://github.com/DanWaxman/Lintel
cd Lintel/src
pip install -e .
```

# Example

For examples, see the code for experiments in the paper under `experiments/`.
