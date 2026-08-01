from pathlib import Path

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

root_dir = Path(__file__).resolve().parents[3]

class Transliteration:
    def __init__(
        self,
        model_name:str="Mavkif/m2m100_rup_ur_to_rur",
        path_to_data:str=f"{root_dir}/results"
    ) -> None:
        pass
