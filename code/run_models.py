import argparse
from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.basic_configs import *

def main(is_production_env):
    if is_production_env:
        # running on Kaggle
        datafolder = '/kaggle/input/ptbxl-original-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
        outputfolder = '/kaggle/working/'
    else:
        datafolder = '../data/ptbxl/'
        outputfolder = '../output/'

    models = [
		conf_stacked_lstm_256_256,
        conf_stacked_lstm_128_128,
        conf_stacked_lstm_64_64,
        conf_stacked_lstm_256_128, 
        conf_stacked_lstm_128_64,
    ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################
    
    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]
    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        print("Preparation done...")
        e.perform()
        print("Performation done...")
        e.evaluate(bootstrap_eval=False)
        print("Evaluation done...")

    utils.generate_ptbxl_summary_table(folder=outputfolder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run attention-neural network models on PTBXL dataset.')
    parser.add_argument('--production', action='store_true')
    args = parser.parse_args()
    main(args.production)