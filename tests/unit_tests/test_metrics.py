# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torchtitan.components.metrics import DeviceMemStats, MetricsProcessor
from torchtitan.tools.utils import NoColor


class TestMetricsProcessor(unittest.TestCase):
    def _build_metrics_processor(self) -> MetricsProcessor:
        metrics_processor = object.__new__(MetricsProcessor)
        metrics_processor.config = SimpleNamespace(log_freq=5)
        metrics_processor.logger = Mock()
        metrics_processor.parallel_dims = SimpleNamespace(non_data_parallel_size=2)
        metrics_processor.device_memory_monitor = Mock()
        metrics_processor.device_memory_monitor.get_peak_stats.return_value = (
            DeviceMemStats(1.0, 10.0, 2.0, 20.0, 0, 0)
        )
        metrics_processor.color = NoColor()
        metrics_processor.gpu_peak_flops = 1e15
        metrics_processor.ntokens_since_last_log = 1000
        metrics_processor.data_loading_times = [0.5, 0.5]
        metrics_processor.time_last_log = 100.0
        metrics_processor.previous_global_avg_loss = 2.0
        metrics_processor.num_flops_per_token = 2000
        metrics_processor.has_quantization = False
        metrics_processor.model_parts = []
        metrics_processor.optimizers = None
        metrics_processor.memory_component_tracker = Mock()
        metrics_processor.memory_component_tracker.build_metrics.return_value = {
            "memory/weights(GiB)": 3.0,
            "memory/activations_peak_estimated(GiB)": 4.0,
            "memory/gradients_peak(GiB)": 5.0,
            "memory/optimizer(GiB)": 6.0,
        }
        return metrics_processor

    @patch("torchtitan.components.metrics.logger.info")
    @patch("torchtitan.components.metrics.time.perf_counter", return_value=110.0)
    def test_log_includes_stat_efficiency_and_goodput(
        self, mock_perf_counter, mock_logger_info
    ):
        metrics_processor = self._build_metrics_processor()

        metrics_processor.log(
            step=5,
            global_avg_loss=1.5,
            global_max_loss=1.7,
            grad_norm=0.25,
        )

        logged_metrics, logged_step = metrics_processor.logger.log.call_args.args
        self.assertEqual(logged_step, 5)
        self.assertAlmostEqual(logged_metrics["throughput(tps)"], 50.0)
        self.assertAlmostEqual(logged_metrics["mfu(%)"], 1e-08)
        self.assertAlmostEqual(logged_metrics["stat_efficiency"], 0.0005)
        self.assertAlmostEqual(logged_metrics["goodput"], 0.025)
        self.assertEqual(logged_metrics["memory/weights(GiB)"], 3.0)
        self.assertEqual(logged_metrics["memory/activations_peak_estimated(GiB)"], 4.0)
        self.assertEqual(logged_metrics["memory/gradients_peak(GiB)"], 5.0)
        self.assertEqual(logged_metrics["memory/optimizer(GiB)"], 6.0)
        self.assertEqual(metrics_processor.ntokens_since_last_log, 0)
        self.assertEqual(metrics_processor.previous_global_avg_loss, 1.5)
        metrics_processor.memory_component_tracker.reset_peak_stats.assert_called_once()
        metrics_processor.device_memory_monitor.reset_peak_stats.assert_called_once()
        mock_perf_counter.assert_called()
        mock_logger_info.assert_called()

    @patch("torchtitan.components.metrics.logger.info")
    @patch("torchtitan.components.metrics.time.perf_counter", return_value=110.0)
    def test_log_skips_stat_efficiency_without_metric(
        self, mock_perf_counter, mock_logger_info
    ):
        metrics_processor = self._build_metrics_processor()
        metrics_processor.previous_global_avg_loss = None

        metrics_processor.log(
            step=5,
            global_avg_loss=1.5,
            global_max_loss=1.7,
            grad_norm=0.25,
        )

        logged_metrics, _ = metrics_processor.logger.log.call_args.args
        self.assertNotIn("stat_efficiency", logged_metrics)
        self.assertNotIn("goodput", logged_metrics)
        mock_perf_counter.assert_called()
        mock_logger_info.assert_called()


if __name__ == "__main__":
    unittest.main()
