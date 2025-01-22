import os
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.inference_engine import TTSInferenceEngine
from tools.llama.generate import launch_thread_safe_queue
from tools.schema import ServeTTSRequest
from tools.vqgan.inference import load_model as load_decoder_model
from tools.webui_ukrainian import build_app
from tools.webui_ukrainian.inference import get_inference_wrapper
from tools.webui_ukrainian.references import scan_references

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="light")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Get project root directory
    project_root = Path(__file__).parent.parent
    references_dir = project_root / "references"
    
    # Scan references directory
    logger.info(f"Project root: {project_root}")
    logger.info(f"Looking for references in: {references_dir}")
    references = scan_references(references_dir)
    
    if not references:
        logger.warning("No references found! Interface will have empty reference selection.")
    
    # Check if MPS or CUDA is available
    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("mps is available, running on mps.")
    elif not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"
    
    # Змінимо на створення device через torch
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")
    
    # Очистимо кеш GPU перед завантаженням моделей
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    # Додамо перевірку, де знаходиться модель
    logger.info(f"Checking model device - Decoder: {next(decoder_model.parameters()).device}")

    # Dry run
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Перевірка.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )

    logger.info("Warming up done, launching the Ukrainian web UI...")

    inference_fct = get_inference_wrapper(inference_engine, references)

    app = build_app(inference_fct, args.theme, references)
    app.launch(show_api=True, server_name='0.0.0.0') 