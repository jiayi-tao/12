from keras.models import load_model
from model import get_testing_model
import os

# 检查文件是否存在
model_path = 'model/keras/model.h5'
if not os.path.exists(model_path):
    print(f"错误：模型文件不存在 - {model_path}")
else:
    print(f"找到模型文件：{model_path}")

# 尝试加载 model.h5 文件
loaded_model = None  # 预先定义变量
try:
    loaded_model = load_model(model_path)
    print("成功加载 model.h5 文件")
    print("加载模型的摘要信息：")
    loaded_model.summary()
except Exception as e:
    print(f"加载 model.h5 文件时出错: {e}")

# 获取 model.py 中定义的模型架构
try:
    defined_model = get_testing_model()
    print("model.py 中定义的模型摘要信息：")
    defined_model.summary()
except Exception as e:
    print(f"获取 model.py 中模型定义时出错: {e}")
    defined_model = None

# 对比两个模型的架构是否一致
if loaded_model and defined_model:
    # 获取两个模型的层信息
    loaded_layers = [(layer.name, layer.__class__.__name__) for layer in loaded_model.layers]
    defined_layers = [(layer.name, layer.__class__.__name__) for layer in defined_model.layers]

    # 打印层信息用于调试
    print("\n加载模型的层结构：")
    for i, (name, type) in enumerate(loaded_layers):
        print(f"{i + 1}. {name} ({type})")

    print("\nmodel.py 定义的层结构：")
    for i, (name, type) in enumerate(defined_layers):
        print(f"{i + 1}. {name} ({type})")

    # 逐层比较
    if loaded_layers == defined_layers:
        print("\n✅ 两个模型的架构完全一致")
    else:
        print("\n❌ 两个模型的架构存在差异")

        # 找出差异
        min_len = min(len(loaded_layers), len(defined_layers))
        for i in range(min_len):
            if loaded_layers[i] != defined_layers[i]:
                print(f"差异位于第 {i + 1} 层:")
                print(f"  加载模型: {loaded_layers[i][0]} ({loaded_layers[i][1]})")
                print(f"  定义模型: {defined_layers[i][0]} ({defined_layers[i][1]})")
                break

        # 检查层数是否不同
        if len(loaded_layers) != len(defined_layers):
            print(f"层数不匹配: 加载模型有 {len(loaded_layers)} 层，定义模型有 {len(defined_layers)} 层")
else:
    print("无法进行模型比较：至少有一个模型未能成功加载或定义")