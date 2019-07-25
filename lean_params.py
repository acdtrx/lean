
gen_params_1M = {
    "ws_label": "1M",
    "ws_size": 1000000,
    "vocab_cutoff": 10
}

gen_params_full = {
    "ws_label": "full",
    "ws_size": 418236956,
    "vocab_cutoff": 40
}

gen_params = gen_params_1M

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
