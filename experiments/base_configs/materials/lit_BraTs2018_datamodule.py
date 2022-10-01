from easydict import EasyDict
import os

config = EasyDict()

config.sr_scale = 8  # 超分辨率参数
config.batch_size = 16  # 批次大小
config.num_workers = min([os.cpu_count(), config.batch_size if config.batch_size > 1 else 0, 8])  # 使用线程数（自动获取）

# config = EasyDict(
#     sr_scale = 8,
#     batch_size = 16,
#     num_workers = 4
# )

# print(config)

"""
更改了线程数的超参数获取方式，提高在不同计算机中运行性能
by : zyh
"""