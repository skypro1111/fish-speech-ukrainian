HEADER_MD = """# Fish Speech

Модель перетворення тексту в мовлення на основі VQ-GAN та Llama, розроблена [Fish Audio](https://fish.audio) та дотренована [@skypro1111](https://huggingface.co/skypro1111) на закритому синтетичному датасеті.

Ви можете знайти оригінальний вихідний код [тут](https://github.com/fishaudio/fish-speech) та оригінальні моделі [тут](https://huggingface.co/fishaudio/fish-speech-1.5).
Адаптований код доступний [тут](https://github.com/skypro1111/fish-speech-ukrainian), адаптована модель [тут](https://huggingface.co/skypro1111/fish-speech-1.5-ukrainian).

Для вербалізації тексту використовується [mbart-large-50-verbalization](https://huggingface.co/skypro1111/mbart-large-50-verbalization).
Для розставлення наголосів [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress).

Код та ваги моделі Fish Speech розповсюджуються за ліцензією CC BY-NC-SA 4.0.

Ми не несемо відповідальності за будь-яке неправильне використання моделі. Будь ласка, враховуйте ваші місцеві закони та правила перед її використанням.
"""

TEXTBOX_PLACEHOLDER = "Введіть ваш текст тут." 