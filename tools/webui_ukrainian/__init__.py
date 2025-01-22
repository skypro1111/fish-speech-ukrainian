from typing import Callable, List
import gradio as gr
from pathlib import Path

from fish_speech.i18n import i18n
from tools.inference_engine.utils import normalize_text
from tools.webui_ukrainian.variables import HEADER_MD, TEXTBOX_PLACEHOLDER
from tools.webui_ukrainian.references import Reference, scan_references, get_reference_dict


def build_app(inference_fct: Callable, theme: str = "light", references: List[Reference] = None) -> gr.Blocks:
    references_dict = get_reference_dict(references)
    reference_names = list(references_dict.keys())
    
    with gr.Blocks(theme=gr.themes.Base()) as app:
        # Створюємо чергу для обробки запитів
        app.queue(default_concurrency_limit=2, max_size=40)  # Збільшуємо ліміти для двох черг
        
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), 
                    placeholder=TEXTBOX_PLACEHOLDER, 
                    lines=10
                )
                
                processed_text = gr.Textbox(
                    label=i18n("Processed Text (with stress marks)"),
                    lines=5,
                    interactive=False,
                )

                with gr.Row():
                    with gr.Column():
                        with gr.Tabs() as tabs:
                            with gr.Tab(label=i18n("Reference Selection")):
                                reference_dropdown = gr.Dropdown(
                                    choices=reference_names,
                                    label=i18n("Select Reference Voice"),
                                    value=None
                                )
                                reference_audio_preview = gr.Audio(
                                    label=i18n("Reference Audio Preview"),
                                    type="filepath",
                                    interactive=False
                                )
                                reference_text_preview = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=2,
                                    interactive=False
                                )

                            with gr.Tab(label=i18n("Custom Reference")):
                                custom_reference_audio = gr.Audio(
                                    label=i18n("Upload Reference Audio"),
                                    type="filepath",
                                    interactive=True
                                )
                                with gr.Row():
                                    transcribe_btn = gr.Button(
                                        value=i18n("Transcribe Audio"),
                                        variant="secondary",
                                        interactive=False
                                    )
                                    clear_text_btn = gr.Button(
                                        value=i18n("Clear Text"),
                                        variant="secondary"
                                    )
                                custom_reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=2,
                                    placeholder=i18n("Enter text manually or use transcription"),
                                    interactive=True
                                )

                            with gr.Tab(label=i18n("Advanced Config")):
                                with gr.Row():
                                    chunk_length = gr.Slider(
                                        label=i18n("Iterative Prompt Length, 0 means off"),
                                        minimum=0,
                                        maximum=300,
                                        value=300,
                                        step=8,
                                    )

                                    max_new_tokens = gr.Slider(
                                        label=i18n(
                                            "Maximum tokens per batch, 0 means no limit"
                                        ),
                                        minimum=0,
                                        maximum=2048,
                                        value=0,
                                        step=8,
                                    )

                                with gr.Row():
                                    top_p = gr.Slider(
                                        label="Top-P",
                                        minimum=0.1,
                                        maximum=0.9,
                                        value=0.7,
                                        step=0.01,
                                    )

                                    repetition_penalty = gr.Slider(
                                        label=i18n("Repetition Penalty"),
                                        minimum=1,
                                        maximum=1.5,
                                        value=1.5,
                                        step=0.01,
                                    )

                                with gr.Row():
                                    temperature = gr.Slider(
                                        label="Temperature",
                                        minimum=0.4,
                                        maximum=0.9,
                                        value=0.8,
                                        step=0.01,
                                    )
                                    seed = gr.Number(
                                        label="Seed",
                                        info="0 means randomized inference, otherwise deterministic",
                                        value=0,
                                    )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001F3A7 " + i18n("Generate"),
                            variant="primary",
                        )
                
                # Приклади текстів під кнопкою Generate
                with gr.Row():
                    gr.Markdown("### " + i18n("Example Texts"))
                
                example_texts = [
                    "Станом на 15.03.2024 року курс валют становить 39,5 грн за 1 USD. НБУ повідомляє про стабілізацію гривні.",
                    "Відстань між Києвом та Львовом складає 542 км. Подорож потягом Інтерсіті №743 триває 5 год 30 хв.",
                    "ЗСУ отримали 25 танків M1 Abrams та 150 БМП. Техніка надійшла в рамках пакету допомоги на суму 2,5 млрд $.",
                    "Температура повітря сьогодні коливається від -5°C вночі до +8°C вдень. Очікується 15 мм опадів.",
                    "У 2023 році ВВП України зріс на 3,5%. ЄБРР прогнозує зростання економіки на 4% у 2024 році."
                ]
                
                for text_example in example_texts:
                    with gr.Row():
                        example_btn = gr.Button(
                            value=text_example[:100] + "..." if len(text_example) > 100 else text_example,
                            size="sm"
                        )
                        example_btn.click(
                            lambda t=text_example: t,
                            outputs=text
                        )

        def update_reference_preview(reference_name):
            if reference_name in references_dict:
                ref = references_dict[reference_name]
                # Очищаємо кастомні поля якщо вибрано референс зі списку
                return str(ref.wav_path), ref.lab_text, None, ""
            return None, "", None, ""

        # Clear reference dropdown when both custom audio and text are provided
        def clear_reference_selection(audio_path, text):
            if audio_path is not None and text.strip():
                return gr.update(value=None)
            return gr.update()

        # Update reference preview when selection changes
        reference_dropdown.change(
            fn=update_reference_preview,
            inputs=[reference_dropdown],
            outputs=[
                reference_audio_preview, 
                reference_text_preview,
                custom_reference_audio,  # Додаємо очищення кастомного аудіо
                custom_reference_text    # Додаємо очищення кастомного тексту
            ]
        )

        # Clear reference selection when custom reference is complete
        custom_reference_audio.change(
            fn=clear_reference_selection,
            inputs=[custom_reference_audio, custom_reference_text],
            outputs=[reference_dropdown]
        )
        custom_reference_text.change(
            fn=clear_reference_selection,
            inputs=[custom_reference_audio, custom_reference_text],
            outputs=[reference_dropdown]
        )

        def update_transcribe_button(audio_path):
            """Активує кнопку транскрипції, якщо є аудіо"""
            return gr.update(interactive=audio_path is not None)

        def clear_text():
            """Очищує текстове поле"""
            return ""

        def transcribe_audio(audio_path):
            """Транскрибує аудіо за допомогою faster-whisper і обробляє текст"""
            if audio_path is None:
                return ""
            
            try:
                from tools.whisper.transcribe import transcribe_audio_file
                from tools.webui_ukrainian.text_processor import TextProcessor

                # Транскрибуємо аудіо
                raw_text = transcribe_audio_file(audio_path)
                if not raw_text:
                    return ""

                # Обробляємо текст через TextProcessor
                text_processor = TextProcessor()
                processed_text = text_processor.process_text(raw_text)
                
                return processed_text
            except Exception as e:
                return ""

        # Оновлюємо стан кнопки транскрипції при зміні аудіо
        custom_reference_audio.change(
            fn=update_transcribe_button,
            inputs=[custom_reference_audio],
            outputs=[transcribe_btn]
        )

        # Додаємо обробник для кнопки транскрипції (окрема черга)
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[custom_reference_audio],
            outputs=[custom_reference_text],
            queue=True,  # Явно вказуємо використання черги
            concurrency_limit=1,  # Окремий ліміт для транскрипції
        )

        # Додаємо обробник для кнопки очищення тексту
        clear_text_btn.click(
            fn=clear_text,
            inputs=[],
            outputs=[custom_reference_text]
        )

        # Submit (окрема черга для інференсу)
        generate.click(
            inference_fct,
            [
                text,
                reference_dropdown,
                custom_reference_audio,
                custom_reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
            ],
            [audio, processed_text, error],
            queue=True,  # Явно вказуємо використання черги
            concurrency_limit=1,  # Окремий ліміт для інференсу
        )

    return app
