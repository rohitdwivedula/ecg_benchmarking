from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.attention_configs import *

def main():
    datafolder = '../data/ptbxl/'
    # datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

    models = [
        # conf_attention_cnn,
        # conf_attention_lstm,
        conf_attention_bilstm
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################
    
    experiments = [
        ('exp0', 'all')
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
        e.evaluate()
        print("Evaluation done...")
    
    # generate greate summary table
    utils.generate_ptbxl_summary_table()

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################

    # e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
    # e.prepare()
    # e.perform()
    # e.evaluate()

    # generate greate summary table
    # utils.ICBEBE_table()

if __name__ == "__main__":
    main()