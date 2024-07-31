import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal",
    "nn": "NbAiLab/nb-wav2vec2-300m-nynorsk",
}


def save_model(language_code, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    processor.save_pretrained(save_dir)

    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.save_pretrained(save_dir)


def download_huggingface_models(output_dir):
    for lang, model_name in DEFAULT_ALIGN_MODELS_HF.items():
        save_dir = os.path.join(output_dir, lang)
        save_model(lang, model_name, save_dir)
        print(f"Downloaded and saved model for language {lang} at {save_dir}")


def download_pytorch_models(output_dir):
    for language, model_name in DEFAULT_ALIGN_MODELS_TORCH.items():
        try:
            save_dir = os.path.join(output_dir, language)
            os.makedirs(save_dir, exist_ok=True)
            bundle = torchaudio.pipelines.__dict__[model_name]
            bundle.get_model(dl_kwargs={"model_dir": save_dir, "file_name": "model.pt"})
            print(f"Downloaded and saved model for language {language} at {save_dir}")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")


if __name__ == "__main__":
    output_dir = "models/align_models"
    os.makedirs(output_dir, exist_ok=True)
    download_huggingface_models(output_dir)
    download_pytorch_models(output_dir)
