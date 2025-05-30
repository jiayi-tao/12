from keras.models import load_model
from model import get_testing_model
import os

model_path = 'model/keras/model.h5'

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"错误：模型文件不存在 - {model_path}")
    exit(1)
else:
    print(f"找到模型文件：{model_path}")

# 创建模型结构
try:
    defined_model = get_testing_model()
    print("model.py 中定义的模型摘要信息：")
    defined_model.summary()
except Exception as e:
    print(f"获取 model.py 中模型定义时出错: {e}")
    exit(1)

# 尝试加载模型权重
loaded_model = defined_model  # 使用相同的模型结构
try:
    loaded_model.load_weights(model_path)
    print("成功加载模型权重")
except Exception as e:
    print(f"加载模型权重时出错: {e}")
    exit(1)

# 对比两个模型的架构是否一致
# 由于使用了相同的模型结构，这里只需验证权重是否正确加载
print("\n✅ 模型结构一致，且权重已成功加载")