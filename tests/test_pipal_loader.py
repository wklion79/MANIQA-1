import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data.PIPAL22.pipal import PIPAL


class PIPALLoaderTests(unittest.TestCase):
    def test_pipal_loader_supports_nested_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dis_dir = root / 'distorted'
            dis_dir.mkdir()

            nested_dir = dis_dir / 'Distortion_1'
            nested_dir.mkdir()

            image_name = 'A0001_00_00.bmp'
            image_path = nested_dir / image_name
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            label_path = root / 'train.txt'
            label_path.write_text(f'{image_name}, 1.0\n')

            dataset = PIPAL(str(dis_dir), str(label_path), transform=None, keep_ratio=1.0)

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertIn('d_img_org', sample)
            self.assertEqual(sample['d_img_org'].shape[0], 3)
            self.assertEqual(sample['score'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
