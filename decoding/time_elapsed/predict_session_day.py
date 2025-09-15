from pathlib import Path
import argparse as ap
import numpy as np
import json
import pickle
from tqdm import tqdm

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported
from utils.data.concatenate import read_and_concate_sessions
from ridge_fns import diagonal_ridge_scores

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--trial_type', type=str, default='animate', help='Trial type. Can be either animate or inanimate.')
    parser.add_argument('--task', type=str, default='visual', help='Task. Can be either visual or memory or visualsubset.')
    parser.add_argument('--nboot', type=int, default=2000, help='Number of bootstraps to use.')

    args = parser.parse_args()
    
    return args

def prepare_data(session_list, triggers, session_days):
    """
    Prepares data for decoding by concatenating sessions from different days. The y array is the session number.

    Parameters
    ----------
    session_list : list
        List of lists of sessions to be concatenated.
    session_days : list
        List of session days. That is, the day of the session relative to the first session.
    triggers : list
        List of triggers to be used.

    Returns
    -------
    X : np.array
        Data array.
    y : np.array
        Label array
    """
    
    for i, sesh in enumerate(session_list):
        sesh = [f'{i}-epo.fif' for i in sesh]
        X, y = read_and_concate_sessions(sesh, triggers)

        # replace all values of y with day of the session as found in session_days
        y = np.array([session_days[i] for _ in range(len(y))])

        if i == 0:
            Xs = X
            ys = y
        else:
            Xs = np.concatenate((Xs, X), axis=1)
            ys = np.concatenate((ys, y), axis=0)

    return Xs, ys

def get_triggers(trial_type:str = "animate"):
    path = Path(__file__)

    with open(path.parents[2] / 'info_files' / 'event_ids.txt', 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    triggers = [value for key, value in event_ids.items() if trial_type.capitalize() in key][:27]

    return triggers

def prep_bootstrap(X, y, n_per_unique_y=200):

    values, counts = np.unique(y, return_counts=True)
    subsample_idx = []

    for val in values:
        class_idx = np.where(y == val)[0]
        sampled_idx = np.random.choice(class_idx, n_per_unique_y, replace=True)
        subsample_idx.extend(sampled_idx)

    subsample_idx = np.array(subsample_idx)

    X_tmp = X[:, subsample_idx, :]
    y_tmp = y[subsample_idx]

    return X_tmp, y_tmp



if __name__ == '__main__':
    args = parse_args()

    path = Path(__file__)
    output_dir= path.parent / "results" / f'{args.trial_type}_{args.task}_session_day'

    # ensure that the results directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'], ['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]
    session_days = [0, 1, 7, 8, 145, 159, 161]
    
    # get the triggers
    triggers = get_triggers(args.trial_type)

    X, y = prepare_data(sessions, triggers, session_days)

    # run bootstraps
    for i in tqdm(range(args.nboot), desc="Running bootstraps"):
        # GET SOME OF THE SAMPLES
        X_tmp, y_tmp = prep_bootstrap(X, y)

        pred, true = diagonal_ridge_scores(X_tmp, y_tmp, cv=5, ncv=1, return_betas=False, alphas = [10])
        output = {
            "predicted": pred, 
            "true": true
        }

        output_file = output_dir / f'bootstrap_{i}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output, f)
