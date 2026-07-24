from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloth06_probe_memory_and_throughput import (
    _recommend,
    _row_configuration,
    _validate_controller_args,
    parse_args as parse_memory_probe_args,
)
from cloth20_probe_dual_gpu_tensor_parallel import (
    _validate_args as validate_tensor_parallel_probe_args,
    _worker_command as tensor_parallel_worker_command,
    network_dimensions as tensor_parallel_network_dimensions,
    parse_args as parse_tensor_parallel_probe_args,
)
from cloth21_train_tensor_parallel_online import (
    next_checkpoint_generation,
    parse_args as parse_tensor_parallel_train_args,
    prune_full_checkpoints,
    resolve_resume_checkpoint,
    run_directory as tensor_parallel_run_directory,
    validate_args as validate_tensor_parallel_train_args,
    worker_command as tensor_parallel_train_worker_command,
)
from tshirt_config import DEFAULT_EVALUATION
from validation_protocol import CHECKPOINT_VALIDATION, FAST_MONITOR, checkpoint_rank


class PipelineContractTests(unittest.TestCase):
    def test_requested_inner_iteration_caps_are_frozen(self) -> None:
        self.assertEqual(FAST_MONITOR.inner_steps, 15)
        self.assertEqual(CHECKPOINT_VALIDATION.inner_steps, 50)
        self.assertEqual(DEFAULT_EVALUATION.convergence_residual_ratio, 1e-3)
        self.assertEqual(DEFAULT_EVALUATION.two_order_single_step_ratio, 1e-2)
        self.assertFalse(FAST_MONITOR.early_stop)
        self.assertFalse(CHECKPOINT_VALIDATION.early_stop)

    def test_checkpoint_rank_is_stability_first(self) -> None:
        base = {
            "failed_motion_count": 0,
            "survival_frame_p05": 100,
            "residual_ratio_p95": 0.1,
            "single_step_le_two_orders_frame_count": 20,
            "energy_increase_fraction": 0.0,
        }
        unstable = {**base, "failed_motion_count": 1, "residual_ratio_p95": 1e-8}
        self.assertLess(checkpoint_rank(base), checkpoint_rank(unstable))

    def test_memory_probe_exposes_model_pool_and_batch_configuration(self) -> None:
        argv = [
            "cloth06_probe_memory_and_throughput.py",
            "--activation", "silu",
            "--depth", "2",
            "--width", "1024",
            "--use-bias",
            "--pool-size", "64",
            "--batch-sizes", "4", "8", "16",
        ]
        with patch("sys.argv", argv):
            args = parse_memory_probe_args()
        _validate_controller_args(args)
        row = _row_configuration(args, batch_size=8)
        self.assertEqual(row["activation"], "silu")
        self.assertEqual(row["depth"], 2)
        self.assertEqual(row["width"], 1024)
        self.assertTrue(row["use_bias"])
        self.assertEqual(row["pool_size"], 64)
        self.assertEqual(row["batch_size"], 8)
        self.assertEqual(row["model_type"], "mlp")
        self.assertIsNone(row["message_passing_steps"])

    def test_memory_probe_exposes_gnn_peak_memory_configuration(self) -> None:
        argv = [
            "cloth06_probe_memory_and_throughput.py",
            "--model-type", "gnn",
            "--activation", "relu",
            "--depth", "2",
            "--width", "128",
            "--no-use-bias",
            "--pool-size", "64",
            "--batch-sizes", "4", "8",
        ]
        with patch("sys.argv", argv):
            args = parse_memory_probe_args()
        _validate_controller_args(args)
        row = _row_configuration(args, batch_size=4)
        self.assertEqual(row["model_type"], "gnn")
        self.assertEqual(row["message_passing_steps"], 5)
        self.assertEqual(row["width"], 128)

    def test_memory_probe_rejects_nonbaseline_gnn_shape(self) -> None:
        argv = [
            "cloth06_probe_memory_and_throughput.py",
            "--model-type", "gnn",
            "--activation", "relu",
            "--depth", "1",
            "--width", "128",
            "--no-use-bias",
        ]
        with patch("sys.argv", argv):
            args = parse_memory_probe_args()
        with self.assertRaisesRegex(ValueError, "GNN baseline requires"):
            _validate_controller_args(args)

    def test_memory_recommendation_preserves_training_shape(self) -> None:
        row = {
            "status": "success",
            "batch_size": 16,
            "pool_size": 64,
            "device": "cuda:0",
            "dtype": "float64",
            "seed": 7,
            "model_type": "gnn",
            "activation": "relu",
            "depth": 2,
            "width": 128,
            "use_bias": False,
            "message_passing_steps": 5,
            "peak_reserved_fraction": 0.5,
            "peak_reserved_gib": 12.0,
            "motions_per_second": 20.0,
        }
        recommendation = _recommend([row], headroom=0.85)
        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        for key in (
            "model_type", "activation", "depth", "width", "use_bias", "pool_size",
            "message_passing_steps",
        ):
            self.assertEqual(recommendation[key], row[key])
        self.assertEqual(recommendation["recommended_batch_size"], row["batch_size"])

    def test_tensor_parallel_probe_defaults_to_width_above_input(self) -> None:
        with patch(
            "sys.argv", ["cloth20_probe_dual_gpu_tensor_parallel.py", "--dry-run"]
        ):
            args = parse_tensor_parallel_probe_args()
        dimensions = tensor_parallel_network_dimensions(num_vertices=4424, width=args.width)
        validate_tensor_parallel_probe_args(args, num_vertices=4424)
        self.assertEqual(dimensions["input_dim"], 39816)
        self.assertEqual(dimensions["width"], 39936)
        self.assertGreater(dimensions["width_to_input_ratio"], 1.0)
        self.assertEqual(dimensions["global_parameter_count"], 2_120_122_368)
        self.assertEqual(dimensions["local_parameter_count"], 1_060_061_184)
        self.assertEqual(args.pool_size, 512)
        self.assertEqual(args.batch_size, 32)
        command = tensor_parallel_worker_command(args)
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=2", command)

    def test_tensor_parallel_probe_rejects_bias(self) -> None:
        with patch(
            "sys.argv",
            ["cloth20_probe_dual_gpu_tensor_parallel.py", "--use-bias", "--dry-run"],
        ):
            args = parse_tensor_parallel_probe_args()
        with self.assertRaisesRegex(ValueError, "no-use-bias"):
            validate_tensor_parallel_probe_args(args, num_vertices=4424)

    def test_tensor_parallel_trainer_defaults_to_measured_configuration(self) -> None:
        args = parse_tensor_parallel_train_args(["--dry-run"])
        validate_tensor_parallel_train_args(args, num_vertices=4424)
        self.assertEqual(args.width, 39936)
        self.assertEqual(args.pool_size, 512)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.dtype, "float32")
        self.assertFalse(args.use_bias)
        command = tensor_parallel_train_worker_command(args)
        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc-per-node=2", command)
        self.assertIn("--worker", command)
        self.assertIn("width_39936_no_bias", str(tensor_parallel_run_directory(args)))

    def test_tensor_parallel_trainer_forwards_resume_and_run_directory(self) -> None:
        args = parse_tensor_parallel_train_args(
            ["--run-dir", "/tmp/tp-contract-run", "--resume", "--dry-run"]
        )
        validate_tensor_parallel_train_args(args, num_vertices=4424)
        command = tensor_parallel_train_worker_command(args)
        self.assertIn("--resume", command)
        self.assertEqual(
            command[command.index("--run-dir") + 1],
            str(Path("/tmp/tp-contract-run").resolve()),
        )

    def test_resume_pointer_rejects_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "checkpoints" / "step_000000100_gen_000000"
            checkpoint.mkdir(parents=True)
            (run_dir / "latest.json").write_text(
                '{"checkpoint":"checkpoints/step_000000100_gen_000000"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                resolve_resume_checkpoint(run_dir)
            (checkpoint / "COMPLETE").touch()
            self.assertEqual(resolve_resume_checkpoint(run_dir), checkpoint.resolve())

    def test_checkpoint_retention_uses_generation_not_rolled_back_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            root = run_dir / "checkpoints"
            names = (
                "step_000000100_gen_000000",
                "step_000000090_gen_000001",
                "step_000000080_gen_000002",
            )
            for name in names:
                path = root / name
                path.mkdir(parents=True)
                (path / "COMPLETE").touch()
            self.assertEqual(next_checkpoint_generation(run_dir), 3)
            prune_full_checkpoints(run_dir, keep=2)
            self.assertFalse((root / names[0]).exists())
            self.assertTrue((root / names[1]).exists())
            self.assertTrue((root / names[2]).exists())


if __name__ == "__main__":
    unittest.main()
