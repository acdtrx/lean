
gen_params_all = {
    "day0-7": {
        "ws_start_time": 0,
        "ws_end_time": 8*86400,
        "vocab_cutoff": 40,
        "csv_filename": './data/auth_users.txt',
        "tensors_filename": './cache/tensors_day0-7.pt',
        "vocab_filename": './cache/vocab_users_day0-7.pickle'
    },
    "day0-7-8": {
        "ws_start_time": 8*86400,
        "ws_end_time": 9*86400,
        "vocab_cutoff": 40,
        "csv_filename": './data/auth_users.txt',
        "tensors_filename": './cache/tensors_day0-7-8.pt',
        "vocab_filename": './cache/vocab_users_day0-7.pickle'
    },
    "day7": {
        "ws_start_time": 7*86400,
        "ws_end_time": 8*86400,
        "vocab_cutoff": 40,
        "csv_filename": './data/auth_users.txt',
        "tensors_filename": './cache/tensors_day7.pt',
        "vocab_filename": './cache/vocab_users_day7.pickle'
    },
    "day7_small": {
        "ws_start_time": 7*86400,
        "ws_end_time": 7*86400+300,
        "vocab_cutoff": 40,
        "csv_filename": './data/auth_users.txt',
        "tensors_filename": './cache/tensors_day7_small.pt',
        "vocab_filename": './cache/vocab_users_day7.pickle'
    },
    "day8": {
        "ws_start_time": 8*86400,
        "ws_end_time": 9*86400,
        "vocab_cutoff": 40,
        "csv_filename": './data/auth_users.txt',
        "tensors_filename": './cache/tensors_day8.pt',
        "vocab_filename": './cache/vocab_users_day7.pickle'
    },
    "redteam7": {
        "ws_start_time": 7*86400,
        "ws_end_time": 8*86400,
        "vocab_cutoff": 40,
        "csv_filename": './cache/auth_red.txt',
        "tensors_filename": './cache/tensors_redteam7.pt',
        "vocab_filename": './cache/vocab_users_day7.pickle'
    },
    "redteam8": {
        "ws_start_time": 8*86400,
        "ws_end_time": 9*86400,
        "vocab_cutoff": 40,
        "csv_filename": './cache/auth_red.txt',
        "tensors_filename": './cache/tensors_redteam8.pt',
        "vocab_filename": './cache/vocab_users_day7.pickle'
    }
}

net_params = {
    "embed_size": 128,
    "hidden_size": 512,
    "bidirectional": True,
    "weights_alpha": 1000
}

trainer_params = {
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 9,
    "starting_epoch": 0,
    "compute_acc_every": 1000
}
