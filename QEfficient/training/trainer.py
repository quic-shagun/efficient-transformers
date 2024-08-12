# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx.inliner
import torch
from onnxruntime import InferenceSession
from onnxruntime.training import api as ort_train_api
from onnxruntime.training import artifacts, onnxblock
from torch.utils.data import Dataset
from transformers import Trainer

from QEfficient.training.onnx_transforms import AddOptimizerTransform, AddTrainingOpsTransform, InputsToInitTransform

from .qaic_infer import QAICInferenceSession

logger = logging.getLogger(__name__)


class QEffTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        if getattr(self, "_wrapped", False):
            return model

        if self.args.bf16 or self.args.bf16_full_eval:
            raise NotImplementedError("BF16 is not currently supported in QEfficient")

        if not self.args.fp16:
            warnings.warn("FP32 training will be very slow and may exceed memory; Try using FP16")

        if not self.args.fp16_full_eval:
            warnings.warn("FP32 evaluation will be very slow and may exceed memory; Try using FP16")

        if self.args.auto_find_batch_size:
            raise NotImplementedError("auto_find_batch_size not supported currently")

        self.trainable_params = set([x for x, p in self.model.named_parameters() if p.requires_grad])
        self.frozen_params = set([x for x, p in self.model.named_parameters() if not p.requires_grad])

        for sample_input in dataloader:
            break

        # Generate artifacts
        self.model_onnx_path = self._export_onnx(sample_input)
        logger.debug("ONNX exported")
        self.train_onnx_path, self.eval_onnx_path = self._generate_artifacts()
        logger.debug("Backward graph generated")

        # Fix, Validate and Save training model
        train_onnx = onnx.load(self.train_onnx_path, load_external_data=False)
        train_onnx, transformed = InputsToInitTransform.apply(train_onnx, self.model_onnx_path, self.frozen_params)
        train_onnx, transformed = AddTrainingOpsTransform.apply(train_onnx)
        if self.args.validate:
            train_onnx_tmp_path = os.path.join(self.args.output_dir, "training_model_tmp.onnx")
            onnx.save(train_onnx, train_onnx_tmp_path)
            self._validate_with_onnxrt(self.train_onnx_path, train_onnx_tmp_path, sample_input, training=True)
            os.remove(train_onnx_tmp_path)
        train_onnx = AddOptimizerTransform.apply(train_onnx)
        train_onnx = onnx.shape_inference.infer_shapes(train_onnx, True, True, True)
        self.train_onnx_path = os.path.join(self.args.output_dir, "training_model_modified.onnx")
        onnx.save(train_onnx, self.train_onnx_path)
        onnx.checker.check_model(self.train_onnx_path, True)
        retained_inputs = {
            x.name[: -len("_RetainedState")] for x in train_onnx.graph.output if x.name.endswith("_RetainedState")
        }
        del train_onnx
        self.custom_io_train_path = os.path.join(self.args.output_dir, "custom_io_train.yaml")
        with open(self.custom_io_train_path, "w") as fp:
            for ioname in retained_inputs:
                fp.write(f" - IOName: {ioname}\n   Precision: float16\n\n")
                fp.write(f" - IOName: {ioname}_RetainedState\n   Precision: float16\n\n")

        # Fix, Validate and Save eval model
        eval_onnx = onnx.load(self.eval_onnx_path, load_external_data=False)
        eval_onnx, transformed = InputsToInitTransform.apply(eval_onnx, self.eval_onnx_path, self.frozen_params)
        eval_onnx, transformed = AddTrainingOpsTransform.apply(eval_onnx)
        if self.args.validate:
            eval_onnx_tmp_path = os.path.join(self.args.output_dir, "eval_model_tmp.onnx")
            onnx.save(eval_onnx, eval_onnx_tmp_path)
            self._validate_with_onnxrt(self.eval_onnx_path, eval_onnx_tmp_path, sample_input, training=False)
            os.remove(eval_onnx_tmp_path)
        eval_onnx = onnx.shape_inference.infer_shapes(eval_onnx, True, True, True)
        self.eval_onnx_path = os.path.join(self.args.output_dir, "eval_model_modified.onnx")
        onnx.save(eval_onnx, self.eval_onnx_path)
        onnx.checker.check_model(self.eval_onnx_path, True)
        assert retained_inputs.issubset({x.name for x in eval_onnx.graph.input})
        del eval_onnx
        self.custom_io_eval_path = os.path.join(self.args.output_dir, "custom_io_eval.yaml")
        with open(self.custom_io_eval_path, "w") as fp:
            for ioname in retained_inputs:
                fp.write(f" - IOName: {ioname}\n   Precision: float16\n\n")

        # self.model_onnx_path = os.path.join(self.args.output_dir, "model.onnx")
        # self.train_onnx_path = os.path.join(self.args.output_dir, "training_model_modified.onnx")
        # self.eval_onnx_path = os.path.join(self.args.output_dir, "eval_model_modified.onnx")
        # self.custom_io_train_path = os.path.join(self.args.output_dir, "custom_io_train.yaml")
        # self.custom_io_eval_path = os.path.join(self.args.output_dir, "custom_io_eval.yaml")

        self._compile_models()

        # self.train_qpc_path = os.path.join(self.args.qpc_path, "train")
        # self.eval_qpc_path = os.path.join(self.args.qpc_path, "eval")

        self._load_models()

        self._wrapped = True
        return model

    def _export_onnx(self, inputs: Dict[str, torch.Tensor]) -> str:
        os.makedirs(self.args.output_dir, exist_ok=True)
        onnx_path = os.path.join(self.args.output_dir, "model.onnx")
        outputs = self.model(**inputs)
        torch.onnx.export(
            self.model,
            (dict(inputs),),
            onnx_path,
            input_names=list(inputs.keys()),
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "labels": {0: "batch_size", 1: "seq_len"},
            },
            output_names=list(outputs.keys()),
            # Support only upto opset 15: https://github.com/microsoft/onnxruntime/pull/19152
            opset_version=15,
            # Following 3 params need to be passed as per ONNXRT-training:
            # https://onnxruntime.ai/docs/api/python/on_device_training/training_artifacts.html
            export_params=True,
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        return onnx_path

    def _generate_artifacts(self) -> Tuple[str, str]:
        onnx_model = onnx.load(self.model_onnx_path, load_external_data=False)

        # Load only trainable params
        for param in onnx_model.graph.initializer:
            if param.name in self.trainable_params and onnx.external_data_helper.uses_external_data(param):
                onnx.external_data_helper.load_external_data_for_tensor(param, self.args.output_dir)
                param.ClearField("data_location")
                param.ClearField("external_data")

        # Check if all parameters are present
        assert (self.trainable_params | self.frozen_params) == {x.name for x in onnx_model.graph.initializer}

        # ONNXRT assumes first output is the loss output
        assert onnx_model.graph.output[0].name == "loss"

        class NoLoss(onnxblock.blocks.PassThrough):
            def __call__(self, *inputs):
                # Fixes onnx checker issue
                return self.build(*inputs)

        artifacts.generate_artifacts(
            onnx_model,
            requires_grad=self.trainable_params,
            frozen_params=self.frozen_params,
            loss=NoLoss(),
            optimizer=artifacts.OptimType.SGD,
            artifact_directory=self.args.output_dir,
        )
        return os.path.join(self.args.output_dir, "training_model.onnx"), os.path.join(
            self.args.output_dir, "eval_model.onnx"
        )

    def _validate_with_onnxrt(
        self, model_orig_path: str, model_fixed_path: str, inputs: Dict[str, torch.Tensor], training: bool
    ):
        inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        if training:
            inputs["lazy_reset_grad"] = np.array([True])

        ckpt = ort_train_api.CheckpointState.load_checkpoint(os.path.join(self.args.output_dir, "checkpoint"))
        for name, param in ckpt.parameters:
            inputs[name] = param.data
            if training:
                inputs[name + "_grad.accumulation.buffer"] = np.zeros_like(param.data)

        session_orig = InferenceSession(model_orig_path)
        session_fixed = InferenceSession(model_fixed_path)
        output_names = [x.name for x in session_orig.get_outputs()]

        outputs = session_fixed.run(None, inputs)
        _ = session_orig.run(None, inputs)

        passed = True
        for name, buffer in inputs.items():
            if name.endswith("accumulation.buffer"):
                diff = np.abs(outputs[output_names.index(name[:-6] + "out")] - buffer).max()
                if diff > 1e-6:
                    passed = False
                    print(name, diff)

        return passed

    def _compile_models(self):
        os.makedirs(self.args.qpc_path, exist_ok=True)
        args = [
            "/opt/qti-aic/exec/qaic-exec",
            "-aic-hw",
            "-aic-hw-version=2.0",
            f"-onnx-define-symbol=batch_size,{self._train_batch_size}",
            f"-onnx-define-symbol=seq_len,{self.args.max_ctx_len}",
            f"-aic-num-cores={self.args.num_cores}",
            "-compile-only",
        ]
        if self.args.fp16:
            args.append("-convert-to-fp16")
        if self.args.mxfp6_matmul:
            args.append("-mxfp6-matmul")
        self.train_qpc_path = os.path.join(self.args.qpc_path, "train")
        subprocess.run(
            [
                *args,
                f"-m={self.train_onnx_path}",
                f"-custom-IO-list-file={self.custom_io_train_path}" if self.args.fp16 else "",
                f"-aic-binary-dir={self.train_qpc_path}",
            ]
        ).check_returncode()
        self.eval_qpc_path = os.path.join(self.args.qpc_path, "eval")
        subprocess.run(
            [
                *args,
                f"-m={self.eval_onnx_path}",
                f"-custom-IO-list-file={self.custom_io_eval_path}" if self.args.fp16 else "",
                f"-aic-binary-dir={self.eval_qpc_path}",
            ]
        ).check_returncode()

    def _load_models(self):
        self.eval_wrapped = QAICInferenceSession(self.eval_qpc_path, self.args.device_ids, activate=False)
        self.model_wrapped = QAICInferenceSession(self.train_qpc_path, self.args.device_ids)

        batch_size, seq_len = self.model_wrapped.bindings[self.model_wrapped.binding_index_map["input_ids"]].dims
        assert batch_size == self._train_batch_size, "Incorrectly compiled qpc"
        assert seq_len == self.args.max_ctx_len, "Incorrectly compiled qpc"

    def training_step(self, inputs):
        # TODO: implement
        raise NotImplementedError()

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # TODO: implement
        raise NotImplementedError()

    def _save_checkpoint(self, model, trial, metrics=None):
        # TODO: implement
        raise NotImplementedError()
