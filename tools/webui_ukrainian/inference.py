import html
from functools import partial
from typing import Any, Callable, List

from fish_speech.i18n import i18n
from tools.schema import ServeReferenceAudio, ServeTTSRequest
from tools.webui_ukrainian.text_processor import TextProcessor
from tools.webui_ukrainian.references import Reference, get_reference_dict

 
def inference_wrapper(
    text,
    reference_name,
    custom_audio,
    custom_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    engine,
    text_processor,
    references_dict,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """
    # Process text through verbalization and stress marking
    processed_text = text_processor.process_text(text)

    # Визначаємо, який референс використовувати
    if custom_audio is not None and custom_text.strip():
        # Використовуємо користувацький референс
        with open(custom_audio, "rb") as audio_file:
            audio_bytes = audio_file.read()
        references = [ServeReferenceAudio(audio=audio_bytes, text=custom_text.strip())]
    elif reference_name in references_dict:
        # Використовуємо вибраний референс зі списку
        reference = references_dict[reference_name]
        with open(reference.wav_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        references = [ServeReferenceAudio(audio=audio_bytes, text=reference.lab_text)]
    else:
        references = []

    req = ServeTTSRequest(
        text=processed_text,
        normalize=False,
        reference_id=None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio, processed_text, None
            case "error":
                return None, None, build_html_error_message(i18n(result.error))
            case _:
                pass

    return None, None, i18n("No audio generated")


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine, references: List[Reference]) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """
    text_processor = TextProcessor()
    references_dict = get_reference_dict(references)
    return partial(
        inference_wrapper,
        engine=engine,
        text_processor=text_processor,
        references_dict=references_dict,
    )
