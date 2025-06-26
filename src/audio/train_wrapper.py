import sys

sys.path.append('src')

from audio.utils.utils import load_config, is_debugging
from audio.train import main as run_simple
from audio.train_multihead import main as run_multihead
from audio.train_multihead_whisper import main as run_multihead_whisper
from audio.train_vae import main as run_vae

if __name__ == "__main__":
    cfg = load_config("audio_config_vae_with_others.yaml")
    debug = is_debugging()
    run_simple(cfg, debug=debug)

    cfg = load_config("audio_config_vae_with_others.yaml")
    debug = is_debugging()
    run_multihead(cfg, debug=debug)

    cfg = load_config("audio_config_vae_with_others.yaml")
    debug = is_debugging()
    run_vae(cfg, debug=debug)

    cfg = load_config("audio_config_vae_with_others.yaml")
    debug = is_debugging()
    run_multihead_whisper(cfg, debug=debug)
