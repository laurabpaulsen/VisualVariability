from pathlib import Path
import pickle

# local imports
from ridge_fns import diagonal_ridge_scores
from predict_session_day import prep_bootstrap, prepare_data, get_triggers, parse_args
from tqdm import tqdm


if __name__ == '__main__':
    args = parse_args()

    path = Path(__file__)
    output_path = path.parent / "results" / f'{args.trial_type}_{args.task}_session_number'

    # ensure that the results directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]
    session_days = [0, 1, 2, 3, 4+4, 5+4, 6+4]
    

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

        output_file = output_path / f'bootstrap_{i}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(output, f)