# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="facebook/nllb-200-3.3B")
print(pipe("你好", src_lang="zh", tgt_lang="en"))