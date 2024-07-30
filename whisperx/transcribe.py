import gc
from typing import Any, List
import torch

from whisperx import vad
from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult

def transcribe_internal(
    audio,
    model_path, 
    vad_model_path,
    audio_language, 
    processingDevice, 
    computeType, 
    numOfThreads, 
    batchSize
) ->  TranscriptionResult | int:
    vad_model = vad.load_vad_model(torch.device(processingDevice), model_fp=vad_model_path)

    model = load_model(
        model_path,
        vad_model=vad_model,
        device=processingDevice,
        compute_type=computeType,
        threads=numOfThreads
    )

    result = model.transcribe(audio, language=audio_language, batch_size=batchSize)

    del model
    gc.collect()
    torch.cuda.empty_cache() 

    return result


def align_transcription_internal(
    audio,
    result,
    alignment_model_path,
    processingDevice
) -> AlignedTranscriptionResult:
    align_model, align_metadata = load_align_model(language_code=result["language"], device=processingDevice, model_path=alignment_model_path)
    res = align(result["segments"], align_model, align_metadata, audio, processingDevice, return_char_alignments=False)

    del align_model
    gc.collect()
    torch.cuda.empty_cache()

    return res


def diarize_transcription_internal(
    audio,
    result,
    pyannoteConfigPath, 
    processingDevice
) -> Any:
    diarize_model = DiarizationPipeline(model_name=pyannoteConfigPath, device=processingDevice)
    diarize_segments = diarize_model(audio)

    return assign_word_speakers(diarize_segments, result)