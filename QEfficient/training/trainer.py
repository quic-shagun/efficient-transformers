# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx.inliner
import torch
from accelerate import skip_first_batches
from onnxruntime import InferenceSession
from onnxruntime.training import api as ort_train_api
from onnxruntime.training import artifacts, onnxblock
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.debug_utils import DebugOption
from transformers.integrations import hp_params
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_utils import HPSearchBackend, TrainOutput, has_length, speed_metrics

from QEfficient.training.onnx_transforms import AddOptimizerTransform, AddTrainingOpsTransform, InputsToInitTransform

from .qaic_infer import QAICInferenceSession

logger = logging.getLogger(__name__)


class QEffTrainer(Trainer):
    def _inner_training_loop(  # noqa: C901
        self,
        batch_size: int,
        args: TrainingArguments = None,
        resume_from_checkpoint: Optional[str] = None,
        trial: Optional[Dict[str, Any]] = None,
        ignore_keys_for_eval=None,
    ):
        if args.bf16 or args.bf16_full_eval:
            raise NotImplementedError("BF16 is not currently supported in QEfficient")

        if not args.fp16:
            warnings.warn("FP32 training will be very slow and may exceed memory; Try using FP16")

        if not args.fp16_full_eval:
            warnings.warn("FP32 evaluation will be very slow and may exceed memory; Try using FP16")

        if args.auto_find_batch_size:
            raise NotImplementedError("auto_find_batch_size not supported currently")

        if args.gradient_accumulation_steps != 1:
            raise NotImplementedError("gradient_accumulation_steps != 1 is not supported currently")

        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            raise NotImplementedError("Underflow/Overflow detection not available in QEfficient")

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in [*self.callback_handler.callbacks, self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # TODO: Is gradient checkpointing possible?

        self._wrap_model(self.model, training=True, dataloader=train_dataloader)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is QAICInferenceSession(Transformers Model)
        # self.eval_wrapped  is QAICInferenceSession(Transformers Model)

        # TODO: Checkpoint loading, should work by default
        if resume_from_checkpoint is not None:
            pass

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += inputs[main_input_name].numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # We run optimizer and training together in this step
                tr_loss_step = self.training_step(inputs)

                if args.logging_nan_inf_filter and (np.isnan(tr_loss_step) or np.isinf(tr_loss_step)):
                    logger.warn("Got inf/nan loss")
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # TODO: Gradient clipping
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, self.model, trial, epoch, ignore_keys_for_eval)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, self.model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _wrap_model(self, model, training=True, dataloader=None):
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
