from __future__ import annotations

import unittest

from cloth5x5.io import resolve_device


class DeviceTests(unittest.TestCase):
    def test_resolve_device_auto(self) -> None:
        device = resolve_device("auto")
        self.assertIn(device.type, {"cpu", "cuda"})

    def test_resolve_device_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
