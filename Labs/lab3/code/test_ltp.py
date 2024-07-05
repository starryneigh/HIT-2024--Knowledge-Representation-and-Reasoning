from ltp import LTP
import os

ltp = LTP("../ltp_small") #加载模型
sentence = "裴友生，男，汉族，湖北蕲春人，1957年12月出生，大专学历。"
#两种任务：先分词、再进⾏命名实体识别
result = ltp.pipeline([sentence], tasks = ["cws","ner"])
print(result.ner)

# path = os.path.join(os.path.dirname(__file__), "data")
# print(path)

# import os

# # 创建一个包含斜杠和反斜杠的路径
# path = 'C:/user/folder\file.txt'

# # 打印路径
# print(path)

# # 使用os.path.join拼接路径
# new_path = os.path.join('C:/user', 'folder/file.txt')

# # 打印新路径
# print(new_path)