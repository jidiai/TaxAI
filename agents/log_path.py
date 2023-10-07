

from pathlib import Path
import os

def make_logpath(algo, n, test=False):
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / Path('./models') / algo / str(n)
    log_dir = base_dir / Path('./models/config_training')
    if not log_dir.exists():
        os.makedirs(log_dir)

    if not model_dir.exists():
        curr_run = 'run1'

    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    os.makedirs(run_dir)

    return run_dir, log_dir