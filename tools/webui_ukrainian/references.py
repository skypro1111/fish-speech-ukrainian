from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from loguru import logger

@dataclass
class Reference:
    name: str  # Назва файлу без розширення
    wav_path: Path
    lab_text: str

def scan_references(references_dir: str | Path) -> List[Reference]:
    """Сканує папку references і повертає список референсів"""
    references_dir = Path(references_dir)
    logger.info(f"Scanning references directory: {references_dir.absolute()}")
    
    if not references_dir.exists():
        logger.warning(f"References directory {references_dir} does not exist")
        return []
    
    references = []
    
    # Знаходимо всі WAV файли
    wav_files = list(references_dir.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files")
    
    for wav_file in wav_files:
        name = wav_file.stem
        lab_file = wav_file.with_suffix('.lab')
        
        logger.info(f"Processing {wav_file.name}")
        
        if lab_file.exists():
            with open(lab_file, 'r', encoding='utf-8') as f:
                lab_text = f.read().strip()
            
            references.append(Reference(
                name=name,
                wav_path=wav_file,
                lab_text=lab_text
            ))
            logger.info(f"Added reference: {name}")
        else:
            logger.warning(f"No .lab file found for {wav_file.name}")
    
    references = sorted(references, key=lambda x: x.name)
    logger.info(f"Total references found: {len(references)}")
    return references

def get_reference_dict(references: List[Reference]) -> Dict[str, Reference]:
    """Створює словник {назва: референс} для швидкого доступу"""
    if references is None:
        logger.warning("References list is None")
        return {}
    return {ref.name: ref for ref in references} 