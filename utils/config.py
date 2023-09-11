import configparser


def read_language_config():
    # 创建配置文件解析器
    config = configparser.ConfigParser()

    # 读取配置文件
    config.read('language.config')

    # 获取配置项的值
    # cnm_value = config['config']['cnm']
    #
    # print(cnm_value)  # 这将输出 "Chinese"
    return config
