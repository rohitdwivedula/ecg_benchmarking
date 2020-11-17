# Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL
This repository is accompanying our article [Deep Learning for ECG Analysis: Benchmarks
and Insights from PTB-XL](https://arxiv.org/abs/2004.13701), which builds on the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6). 
It allows to reproduce the ECG benchmarking experiments described in the paper and to benchmark
user-provided models within our framework. We also maintain a leaderboard for the described PTB-XL dataset
on this page, so feel free to submit your results as PRs.

Please acknowledge our work by citing the corresponding articles listed in **References** below.


## Setup

### Install dependencies
Install the dependencies by creating a conda environment:

    conda env create -f ecg_env.yml
    conda activate ecg_python37

### Get data
Download and prepare the datasets (PTB-XL and ICBEB) via the follwing bash-script:

    ./get_datasets.sh

This script first downloads [PTB-XL from PhysioNet](https://physionet.org/content/ptb-xl/) and stores it in `data/ptbxl/`. 
Afterwards all training data from the [ICBEB challenge 2018](http://2018.icbeb.org/Challenge.html) is downloaded and temporally stored in `tmp_data/`. 
After downloading and unzipping `code/utils/convert_ICBEB.py` is called which stores the data in appropriate format in `data/ICBEB/`. 

## Reproduce results from the paper

Change directory: `cd code` and then call

    python reproduce_results.py

This will perform all experiments for all models used in the paper. 
Depending on the executing environment, this will take up to several hours. 
Once finished, all trained models, predictions and results are stored in `output/`, 
where for each experiment a sub-folder is created each with `data/`, `models/` and `results/` sub-sub-folders. 

### Download models and results

We also provide a [compressed zip-archive](https://datacloud.hhi.fraunhofer.de/nextcloud/s/gLkjQL94d7FXBbS) containing the `output` folder corresponding to our runs including trained models and predictions from our runs mentioned in the leaderboard below. 

## Benchmarking user-provided models
For creating custom benchmarking results our recommendation is as follows:

1. create your model `code/models/your_model.py` which implements a standard classifier interface with `fit(X_train, y_train, X_val, y_val)` and `predict(X)`
2. create a config file `code/configs/your_configs.py` with name, type and parameters (if needed)
3. add your modeltype and model import to the cases in `perform`-function of `code/experiments/scp_experiment.py` (already added for demonstration purpose!)
4. add your model-config to `models` and perform your experiment as below (adjusted code of `code/reproduce_results.py`):

```python
from experiments.scp_experiment import SCP_Experiment
from configs.your_custom_configs import your_custom_config

datafolder = '../data/ptbxl/'
outputfolder = '../output/'

models = [your_custom_config]

e = SCP_Experiment('your_custom_experiment', 'diagnostic', datafolder, outputfolder, models)
e.prepare()
e.perform()
e.evaluate()
```

### Notes on e.evaluate()
Altough we recommend to use our framework, custom evaluation of custom models is still possible via calling `code.utils.utils.evaluate_experiment(y_true, y_pred, thresholds)` 
manually with classwise thresholds. 

For `e.evaluate()`: If the name of the experiment is `exp_ICBEB` classifier thresholds are needed. 
In any other case `evaluate_experiment(y_true, y_pred)` will return a dictionary with `macro_auc` and `Fmax` (both metrics are **without any explicitly needed thresholds**). 
In case of `exp_ICBEB` we offer two functions for computing thresholds (located in `code/utils/utils.py`):

1. `thresholds = utils.find_optimal_cutoff_thresholds(y_train, y_train_pred)`
2. `thresholds = utils.find_optimal_cutoff_thresholds_for_Gbeta(y_train, y_train_pred)`

In addition to `macro_auc` and `Fmax` `evaluate_experiment(y_true, y_pred, thresholds)` will return `F_beta_macro` and `G_beta_macro` as proposed in the physionet-challenge.

### Notes on bootstrapping
Since bootstrapping results might take a while (even in parallel as in our code), we offer a flag for evaluation `e.evaluate(bootstrap_eval=False)` which just performs one single whole sample evaluation. 

**If you want to bootstrap your results:** In each respective experiment-folder `output/exp_*/` the bootstrapping ids for training, 
testing and validation is stored as numpy-arrays containing lists of ids. Otherwise create manually with `utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples)`. For sequential evaluation of those ids, the code might look like:

```python
if experiment_name == 'exp_ICBEB':
    thresholds = utils.find_optimal_cutoff_thresholds(y_train, y_train_pred)
else:
    thresholds = None

train_bootstrap_samples = np.array(utils.get_appropriate_bootstrap_samples(y_train, n_bootstraping_samples))
tr_df = pd.concat([utils.evaluate_experiment(y_train[ids], y_train_pred[ids], thresholds) for ids in train_bootstrap_samples])

tr_df.quantile(0.05), tr_df.mean(), tr_df.quantile(0.95)
```

### Notes on Finetuning
In [this jupyter notebook](https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/Finetuning-Example.ipynb) we provide a basic example of how to finetune our provided models on your custom dataset.



## Leaderboard

We encourage other authors to share their results on this dataset by submitting a PR. The evaluation proceeds as described in the manuscripts: 
In all cases we restrict to classes with more than 50 entries in the whole dataset. The reported scores are test set scores (fold 10) as output 
of the above evaluation procedure and should **not be used for hyperparameter tuning or model selection**. In the provided code, we use folds 1-8 for training, 
fold 9 as validation set and fold 10 as test set. We encourage to submit also the prediction results (`preds`, `targs`, `classes` saved as numpy arrays `preds_x.npy` 
and `targs_x.npy` and `classes_x.npy`) to ensure full reproducibility and to make source code and/or pretrained models available.

 ### 1. PTB-XL: all statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 
| inception1d | 0.925(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| xresnet1d101 | 0.925(07) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| resnet1d_wang | 0.919(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.918(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.914(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.907(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.849(13) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 

 ### 2. PTB-XL: diagnostic statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 
| xresnet1d101 | 0.937(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| resnet1d_wang | 0.936(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.932(07) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| inception1d | 0.931(09) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.927(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.926(10) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.855(15) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 

 ### 3. PTB-XL: Diagnostic subclasses 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 
| inception1d | 0.930(10) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| xresnet1d101 | 0.929(14) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.928(10) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| resnet1d_wang | 0.928(10) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.927(11) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.923(12) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.859(16) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 

 ### 4. PTB-XL: Diagnostic superclasses 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 
| resnet1d_wang | 0.930(05) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| xresnet1d101 | 0.928(05) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.927(05) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.925(06) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| inception1d | 0.921(06) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.921(06) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.874(07) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 

 ### 5. PTB-XL: Form statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 
| inception1d | 0.899(22) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| xresnet1d101 | 0.896(12) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| resnet1d_wang | 0.880(15) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.876(15) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.869(12) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.851(15) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.757(29) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 

 ### 6. PTB-XL: Rhythm statements 
 
| Model | AUC &darr; | paper/source | code | 
|---:|:---|:---|:---| 
| xresnet1d101 | 0.957(19) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| inception1d | 0.953(13) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.953(09) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.949(11) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| resnet1d_wang | 0.946(10) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.931(08) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.890(24) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 


### 7. ICBEB: All statements

| Model | AUC &darr; |  F_beta=2 | G_beta=2 | paper/source | code | 
|---:|:---|:---|:---|:---|:---| 
| xresnet1d101 | 0.974(05) | 0.819(30) | 0.602(37) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| resnet1d_wang | 0.969(06) | 0.803(31) | 0.586(37) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm | 0.964(06) | 0.790(31) | 0.561(37) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| inception1d | 0.963(09) | 0.807(30) | 0.594(41) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| lstm_bidir | 0.959(11) | 0.796(31) | 0.573(36) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| fcn_wang | 0.957(08) | 0.787(31) | 0.563(37) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 
| Wavelet+NN | 0.905(14) | 0.665(34) | 0.405(36) | [our work](https://arxiv.org/abs/2004.13701) | [this repo](https://github.com/helme/ecg_ptbxl_benchmarking/)| 

# References
Please acknowledge our work by citing our preprint

    @article{Strodthoff2020:ecgbenchmarking,
    title={Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL},
    author={Nils Strodthoff and Patrick Wagner and Tobias Schaeffter and Wojciech Samek},
    journal={arXiv preprint 2004.13701},
    year={2020},
    eprint={2004.13701},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    }
	
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