# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

from torchtitan.tools.utils import get_peak_flops


class TestGetPeakFlops(unittest.TestCase):
    @patch("torchtitan.tools.utils.subprocess.run", side_effect=FileNotFoundError())
    def test_rtx_5000_ada_supported(self, mock_run):
        self.assertEqual(get_peak_flops("NVIDIA RTX 5000 Ada Generation"), 261.2e12)
        mock_run.assert_called_once_with(["lspci"], stdout=-1, text=True)


if __name__ == "__main__":
    unittest.main()
