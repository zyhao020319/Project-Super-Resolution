from experiments.dataset_configs.brats2018slice_configs import BraTs2018SliceConfigs
from experiments.method_configs.voxelmorph import VoxelMorphConfigs
from experiments.experiment_configs import ExperimentConfigs


exp = ExperimentConfigs(exp_name="voxelmorph")
datamodule = BraTs2018SliceConfigs()
model = VoxelMorphConfigs()

datamodule.batch_size = 16
model.vis.batch_size = datamodule.batch_size