from easydict import EasyDict


class ISSRNetConfigs:
    def __init__(self):
        # ----------------------------------------------#
        # 形变场可视化的相关参数
        # ----------------------------------------------#
        self.vis = EasyDict(
            size=(240, 240),
            path="results/grid.png",
            grid_density=10,
            batch_size=16,
            step=100
        )
        # ----------------------------------------------#
        # slice interpolation net的参数
        # ----------------------------------------------#
        self.sin = EasyDict(
            in_channels=2,
            out_channels=2,
            conv_kernel_size=(3, 3),
            ngf=32,
            nlayers=4,
            bilinear=True,
            input_shape=(240, 240)
        )

        # ----------------------------------------------#
        # Adam的参数
        # ----------------------------------------------#
        self.adam_optimizer = EasyDict(
            lr=1e-3,
            beta1=0.5,
            beta2=0.99
        )

        # ----------------------------------------------#
        # plateau sheduler的参数
        # ----------------------------------------------#
        self.plateau_scheduler = EasyDict(
            mode="min",
            factor=0.2,
            patience=5
        )

        # ----------------------------------------------#
        # 损失函数的权重调整
        # ----------------------------------------------#
        self.loss = EasyDict(
            shape=(1, 240, 240),  # 用于计算ncc loss，要计算的两个图像的维度，batch size那个维度不要
            length=9,  # 用于计算smoothness loss，平滑滤波器的核大小
            alpha=0.01,  # 用于计算最终loss，平衡ncc loss核smoothness loss，乘在smoothness loss前面
            penalty="l2"  # 用于计算smoothness loss，使用l1 penalty还是l2 penalty
        )
