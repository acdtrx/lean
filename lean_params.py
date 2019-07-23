
gen_params = {
    "ws_label": "1M",
    "ws_size": 1000000,
    # "ws_size": 418236956
}

net_params = {
    "embed_size": 128,
    "hidden_size": 512,
    "bidirectional": True,
    "weights_alpha": 10000
}

trainer_params = {
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 4,
    "compute_acc_every": 10
}
