from app.services.models import model_dir

def load_custom_model_from_hf(repo_id, model_filename='pytorch_model.bin', config_filename=None):
    path=model_dir(repo_id)/model_filename
    if not path.is_file(): raise FileNotFoundError(f'Download models first: {path}')
    return str(path) if config_filename is None else (str(path), str(model_dir(repo_id)/config_filename))
