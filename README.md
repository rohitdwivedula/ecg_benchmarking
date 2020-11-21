 [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 
# Deep Learning Models for ECG Analysis: PTB-XL (2020)

This repository, contains code and results of running various deep learning models on the PTB-XL dataset.

## Setup

### Install dependencies
Install the dependencies by creating a conda environment:

    conda env create -f ecg_env.yml
    conda activate ecg_python37

### Data

The dataset used for this experimentation can be found on PhysioNet: [PTB-XL from PhysioNet](https://physionet.org/content/ptb-xl/). The datasets can be downloaded by running the follwing bash-script: 

    ./get_datasets.sh

This script first downloads  and stores it in `data/ptbxl/`.

## Run Attention Models

Change directory: `cd code` and then call

    python run_attention.py

This will perform all experiments for all the attention based models and save the results. Once finished, all trained models, predictions and results are stored in `output/`, where for each experiment a sub-folder is created each with `data/`, `models/` and `results/` sub-sub-folders. 

## Results

 ### 1. PTB-XL: all statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 

 ### 2. PTB-XL: diagnostic statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 

 ### 3. PTB-XL: Diagnostic subclasses 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 

 ### 4. PTB-XL: Diagnostic superclasses 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 

 ### 5. PTB-XL: Form statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 

 ### 6. PTB-XL: Rhythm statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 


### 7. ICBEB: All statements

| Model | AUC &darr; |  F_beta=2 | G_beta=2 | paper/source | code | 
|---:|:---|:---|:---|:---|:---| 

# References
	
For the PTB-XL dataset, please cite

    @article{Wagner2020:ptbxl,
    author={Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Dieter Kreiseler and Fatima I. Lunze and Wojciech Samek and Tobias Schaeffter},
    title={{PTB-XL}, a large publicly available electrocardiography dataset},
    journal={Scientific Data},
    year={2020},
    note={awaiting publication}
    }

    @misc{Wagner2020:ptbxlphysionet,
    title={{PTB-XL, a large publicly available electrocardiography dataset}},
    author={Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Wojciech Samek and Tobias Schaeffter},
    doi={10.13026/qgmg-0d46},
    year={2020},
    journal={PhysioNet}
    }

    @article{Goldberger2020:physionet,
    author = {Ary L. Goldberger  and Luis A. N. Amaral  and Leon Glass  and Jeffrey M. Hausdorff  and Plamen Ch. Ivanov  and Roger G. Mark  and Joseph E. Mietus  and George B. Moody  and Chung-Kang Peng  and H. Eugene Stanley },
    title = {{PhysioBank, PhysioToolkit, and PhysioNet}},
    journal = {Circulation},
    volume = {101},
    number = {23},
    pages = {e215-e220},
    year = {2000},
    doi = {10.1161/01.CIR.101.23.e215}
    }

If you use the [ICBEB challenge 2018 dataset](http://2018.icbeb.org/Challenge.html) please acknowledge

    @article{liu2018:icbeb,
    doi = {10.1166/jmihi.2018.2442},
    year = {2018},
    month = sep,
    publisher = {American Scientific Publishers},
    volume = {8},
    number = {7},
    pages = {1368--1373},
    author = {Feifei Liu and Chengyu Liu and Lina Zhao and Xiangyu Zhang and Xiaoling Wu and Xiaoyan Xu and Yulin Liu and Caiyun Ma and Shoushui Wei and Zhiqiang He and Jianqing Li and Eddie Ng Yin Kwee},
    title = {{An Open Access Database for Evaluating the Algorithms of Electrocardiogram Rhythm and Morphology Abnormality Detection}},
    journal = {Journal of Medical Imaging and Health Informatics}
    }

Code template for reading and preprocessing data taken from the work done by:

```
@article{Strodthoff2020:ecgbenchmarking,
    title={Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL},
    author={Nils Strodthoff and Patrick Wagner and Tobias Schaeffter and Wojciech Samek},
    journal={arXiv preprint 2004.13701},
    year={2020},
    eprint={2004.13701},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    }
```
