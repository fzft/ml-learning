import os


def get_config():
    return {
        "batch_size": 8,
        "seq_len": 350,
        "d_model": 512,
        "num_epochs": 20,
        "lr": 1e-4,
        "src_lang": "en",
        "tgt_lang": "it",
        "model_folder": "weights",
        "model_file": "transformer_{0}_{1}.pt",
        "preload" : None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "transformer",
        "model_basename": "tmodel_",
    }

def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join('.', model_folder, model_filename)
