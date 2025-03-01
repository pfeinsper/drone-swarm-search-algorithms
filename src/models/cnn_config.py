from dataclasses import dataclass, field


@dataclass
class CNNConfig:
    kernel_sizes: list = field(default_factory=lambda : [(3, 3), (2, 2)])
    out_channels: list = field(default_factory=lambda : [16, 32])

    def get_flatten_size(self, obs_space_shape):
        dim_0 = obs_space_shape[0]
        dim_1 = obs_space_shape[1]
        for kernel_size in self.kernel_sizes:
            dim_0 = (dim_0 - kernel_size[0]) + 1
            dim_1 = (dim_1 - kernel_size[1]) + 1
        return self.out_channels[-1] * dim_0 * dim_1
