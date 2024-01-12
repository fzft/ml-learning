import os
from huggingface_hub import snapshot_download

local_root_model_dir = '/mnt/sda/huggingface/data/models'
local_root_cache_dir = '/mnt/sda/huggingface/data/cache'

token = os.environ["HF_TOKEN"]

# Replace this if you want to use a different model
model_id = "mistralai/Mistral-7B-v0.1"
path_model = os.path.join(local_root_model_dir, model_id)
path_cache = os.path.join(local_root_cache_dir, model_id)

# create a directory to store the model
os.makedirs(path_model, exist_ok=True)

# create a directory to store the cache
os.makedirs(path_cache, exist_ok=True)
snapshot_download(repo_id=model_id, local_dir=path_model, cache_dir=path_cache, proxies={
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}, token=token)
