# Replicating SimGCD

# 项目结构
'''
gcd-task/
|-- analysis/           # 使用notebook做代码测试
|-- data/               # 数据集文件夹
|-- experiment/         # 存放每次实验结果（log、model、tensorboard_log）的文件夹
|-- models/             # 存放模型定义的文件夹
|-- pretrained/         # 存放DINO预训练模型权重的文件夹
|-- utils/              # 存放损失函数和训练工具的文件夹
|-- dataset.py          # 数据集读取、分割脚本
|-- main.py             # 主训练和测试脚本
|-- requirements.txt    # 项目依赖列表
|-- README.md           # 项目描述和使用说明
'''

# 测试代码
python main.py --evaluate True

# 运行代码
python main.py --experiment_name "xxx"

# 日志、模型文件
'''
log: experiment/08-26-22-57-Experiment-2/log/log.txt
model: experiment/08-26-22-57-Experiment-2/models/model.pth
tensorboard_log：experiment/08-26-22-57-Experiment-2/tensorboard_log
'''