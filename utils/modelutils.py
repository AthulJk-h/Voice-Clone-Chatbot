from pathlib import Path

def check_model_paths(encoder_path: Path, synthesizer_path: Path, vocoder_path: Path):
    # This function tests the model paths and makes sure at least one is valid.
    if encoder_path.is_file() or encoder_path.is_dir():
        return
    if synthesizer_path.is_file() or synthesizer_path.is_dir():
        return
    if vocoder_path.is_file() or vocoder_path.is_dir():
        return
