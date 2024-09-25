# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestLayerNormalizeReluXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["layer_norm_relu_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=50))
        channel = 128
        x_shape = [batch_size, channel]
        y_shape = x_shape

        epsilon = draw(st.floats(min_value=0.0000001, max_value=0.001))
        bias = draw(st.floats(min_value=0.0000001, max_value=0.001))
        # Here we will compose a program
        layer_norm_op = OpConfig(
            type='layer_norm',
            inputs={
                'X': ['layer_norm_X'],
                'Bias': ['layer_norm_Bias'],
                'Scale': ['layer_norm_Scale'],
            },
            outputs={
                'Y': ['layer_norm_Y'],
                'Mean': ['layer_norm_Mean'],
                'Variance': ['layer_norm_Variance'],
            },
            epsilon=epsilon,
        )
        relu_op = OpConfig(
            "relu",
            inputs={
                "X": ["layer_norm_Y"],
            },
            outputs={
                "Out": ["relu_Out"],
            },
        )
        mini_graph = [layer_norm_op, relu_op]

        program_config = ProgramConfig(
            ops=mini_graph,
            weights={
                "layer_norm_Scale": TensorConfig(shape=[x_shape[1]]),
                "layer_norm_Bias": TensorConfig(shape=[x_shape[1]]),
            },
            inputs={
                "layer_norm_X": TensorConfig(shape=x_shape),
            },
            outputs=mini_graph[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["layer_norm_relu_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
