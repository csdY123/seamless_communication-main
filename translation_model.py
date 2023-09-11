import argparse
import logging
import torch
import torchaudio
from seamless_communication.models.inference import Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

def call(inputsentence,task,tgt_lang,src_lang,translator):

    output_path="./"
    model_name="seamlessM4T_large"
    vocoder_name="vocoder_36langs"
    ngram_filtering=False

    if task.upper() in {"S2ST", "T2ST"} and output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")

    translated_text, wav, sr = translator.predict(
        inputsentence,
        task,
        tgt_lang,
        src_lang,
        ngram_filtering,
    )

    if wav is not None and sr is not None:
        logger.info(f"Saving translated audio in {tgt_lang}")
        torchaudio.save(
            output_path,
            wav[0].cpu(),
            sample_rate=sr,
        )
    logger.info(f"Translated text in {tgt_lang}: {translated_text}")
    return f"Translated text in {tgt_lang}: {translated_text}"