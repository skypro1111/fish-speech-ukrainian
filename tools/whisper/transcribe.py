from faster_whisper import WhisperModel
from loguru import logger
import torch


_model = None

def get_whisper_model():
    global _model
    if _model is None:
        logger.info("Loading Whisper model...")
        _model = WhisperModel(
            model_size_or_path="turbo",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
            download_root="checkpoints/whisper"
        )
        logger.info("Whisper model loaded")
    return _model

def transcribe_audio_file(audio_path: str) -> str:
    """
    Транскрибує аудіофайл за допомогою faster-whisper
    """
    model = get_whisper_model()
    
    try:
        # Транскрибуємо аудіо
        segments, _ = model.transcribe(
            audio_path,
            language="uk",
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt="Це транскрипція українською мовою."
        )
        
        # Збираємо текст з усіх сегментів
        text = " ".join([segment.text for segment in segments]).strip()
        
        logger.info(f"Transcribed text: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise 