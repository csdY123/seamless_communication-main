import argparse
import logging
import torch
import torchaudio
from seamless_communication.models.inference import Translator
import time
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

def call(inputsentence,task,tgt_lang,src_lang,translator):

    output_path="./"
    model_name="seamlessM4T_large" # seamlessM4T_medium
    vocoder_name="vocoder_36langs"
    ngram_filtering=False

    if task.upper() in {"S2ST", "T2ST"} and output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")
    print("_______________________________________")
    start_time = time.time()
    #for i in range(100):
    translated_text, wav, sr = translator.predict(
        inputsentence,
        task,
        tgt_lang,
        src_lang,
        ngram_filtering,
    )
    end_time = time.time()
    run_time = end_time - start_time
    print(f"函数运行时间：{run_time:.6f} 秒")
    print("_______________________________________")
    if wav is not None and sr is not None:
        logger.info(f"Saving translated audio in {tgt_lang}")
        torchaudio.save(
            output_path,
            wav[0].cpu(),
            sample_rate=sr,
        )
    logger.info(f"Translated text in {tgt_lang}: {translated_text}")
    return f"Translated text in {tgt_lang}: {translated_text}"