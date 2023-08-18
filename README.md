# Replicating SimGCD

# 项目结构
gcd-task/
|-- data/               # 数据集文件夹
|-- models/             # 存放模型定义的文件夹
|-- checkpoints/        # 存放训练模型权重的文件夹
|-- utils/              # 存放辅助函数和工具的文件夹
|-- logs/               # 存放训练日志的文件夹
|-- tensorboard_logs/   # 存放TensorBoard日志的文件夹
|-- pretrained/         # 存放预训练模型权重的文件夹
|-- experiments/        # 存放每次实验的配置和结果
|-- main.py             # 主训练和测试脚本
|-- evaluate.py         # 评估模型性能的脚本
|-- config.py           # 存放默认项目配置（如超参数）的脚本
|-- requirements.txt    # 项目依赖列表
|-- README.md           # 项目描述和使用说明
