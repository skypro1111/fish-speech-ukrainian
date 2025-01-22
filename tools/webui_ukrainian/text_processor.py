import warnings
from transformers import MBartForConditionalGeneration, AutoTokenizer
from ukrainian_word_stress import Stressifier, StressSymbol
import torch

# Ігноруємо попередження про автокаст
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
# Ігноруємо попередження про pickle protocol
warnings.filterwarnings("ignore", category=UserWarning, module="torch._weights_only_unpickler")
# Ігноруємо попередження про старий формат pretrain
warnings.filterwarnings("ignore", category=UserWarning, module="stanza.models.common.pretrain")

class TextProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # Додамо для діагностики
        
        # Initialize verbalization model
        model_name = "skypro1111/mbart-large-50-verbalization"
        
        # Використовуємо новий синтаксис для autocast
        self.verb_model = MBartForConditionalGeneration.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            device_map="auto",  # Це дозволить transformers автоматично розмістити модель
            torch_dtype=torch.float16  # Використовуємо half precision для економії пам'яті
        )
        self.verb_model.eval()
        
        # Перевіряємо, чи модель дійсно на GPU
        print(f"Model device: {next(self.verb_model.parameters()).device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = "uk_XX"
        self.tokenizer.tgt_lang = "uk_XX"
        
        # Initialize stress model
        self.stressify = Stressifier(stress_symbol="ˈ")
        
    def process_text(self, text: str) -> str:
        # Verbalize text
        input_text = f"<verbalization>:{text}"
        
        # Токенізуємо текст
        encoded_input = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        )
        
        # Переміщуємо вхідні дані на GPU
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Генеруємо текст
        with torch.no_grad():
            try:
                output_ids = self.verb_model.generate(
                    **encoded_input, 
                    max_length=1024, 
                    num_beams=5, 
                    early_stopping=True
                )
                verbalized_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Error during text generation: {e}")
                verbalized_text = text  # Повертаємо оригінальний текст у випадку помилки
        
        # Add stress marks and return final processed text
        return self.stressify(verbalized_text.replace("ʼ", "")) 