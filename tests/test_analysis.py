import unittest

from h3_transfer_boost.entropy import entropy_bits_per_byte, estimated_ratio


class EntropyTests(unittest.TestCase):
    def test_entropy_extremes(self):
        self.assertEqual(entropy_bits_per_byte([7] * 1024), 0.0)
        entropy = entropy_bits_per_byte(list(range(256)) * 4)
        self.assertAlmostEqual(entropy, 8.0)
        self.assertEqual(estimated_ratio(entropy), 1.0)

    def test_entropy_midrange(self):
        entropy = entropy_bits_per_byte(list(range(16)) * 64)
        self.assertAlmostEqual(entropy, 4.0)
        self.assertEqual(estimated_ratio(entropy), 0.5)


if __name__ == "__main__":
    unittest.main()
