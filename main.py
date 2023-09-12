# -- coding: UTF-8 --
import torch
from flask import Flask, request,jsonify
from utils.config import read_language_config
from translation_model import call
import logging
from seamless_communication.models.inference import Translator

app = Flask(__name__)
logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
    logger.info(f"Running inference on the GPU in {dtype}.")
else:
    device = torch.device("cpu")
    dtype = torch.float32
    logger.info(f"Running inference on the CPU in {dtype}.")

translator = Translator("seamlessM4T_large", "vocoder_36langs", device, dtype)#seamlessM4T_large seamlessM4T_medium


@app.route('/')
def hello_world():
    return 'hello world'


@app.route('/translate', methods=['POST'])
def mess():  # 我正在使用python写代码，目前有这么一个需求，sentence, sourceLanguage, targetLanguage
    # 从HTTP请求中获取参数
    sentence = request.json.get('sentence')
    sourceLanguage = request.json.get('sourceLanguage')
    targetLanguage = request.json.get('targetLanguage')
    model = request.json.get('languageModel')

    modelResponse=modelCsd(sentence,sourceLanguage,targetLanguage)
    return jsonify({
            "model": "seamlessM4T_large",
            "vocoder": "vocoder_36langs",
            "sentence": sentence,
            "sourceLanguage": sourceLanguage,
            "targetLanguage": targetLanguage,
            "modelResponse":modelResponse
        }), 200

def modelCsd(sentence,sourceLanguage,targetLanguage):
    return call(sentence, "t2tt", targetLanguage, sourceLanguage,translator)

# ./scripts/m4t/predict/predict.py "你好" t2tt en --src_lang zh

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="13144")
    app.run(threaded=True)  # 开启多线程

# if __name__ == '__main__':
#     print(read_language_config()['configTest']['cnm'])
#     mess("你好","cmn","eng")
#     mess("我是陈森达","cmn","eng")
