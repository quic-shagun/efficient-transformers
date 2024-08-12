# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional, Set, Tuple

import onnx
from onnx import ModelProto

from QEfficient.base.onnx_transforms import OnnxTransform


class InputsToInitTransform(OnnxTransform):
    @classmethod
    def apply(
        cls, model: ModelProto, reference_model_path: str, input_names: Set, onnx_base_dir: Optional[str] = None
    ) -> Tuple[ModelProto, bool]:
        reference_model = onnx.load(reference_model_path, load_external_data=False)
        initializers = {init.name: init for init in reference_model.graph.initializer}

        assert input_names.issubset(
            {x.name for x in model.graph.input}
        ), "Some input_names missing as inputs in backward model"
        assert input_names.issubset(initializers.keys()), "input_names missing as initializers in the reference model"

        inputs = []
        transformed = False
        for inp in model.graph.input:
            if inp.name in input_names:
                transformed = True
                init = onnx.TensorProto()
                init.CopyFrom(initializers[inp.name])
                model.graph.initializer.append(init)
            else:
                inputs.append(inp)

        if transformed:
            del model.graph.input[:]
            model.graph.input.extend(inputs)

        return model, transformed
