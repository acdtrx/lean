
gen_params_day7 = {
    "ws_label": "day7",
    "ws_start_time": 7*86400,
    "ws_end_time": 8*86400,
    "vocab_cutoff": 40
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
    "epochs": 4,
    "compute_acc_every": 100
}
