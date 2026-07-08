import shutil
import tempfile
import unittest
from pathlib import Path

from data.PIPAL22.prepare_train_dis import prepare_train_dis


class PrepareTrainDisTests(unittest.TestCase):
    def test_prepare_train_dis_copies_only_labelled_distortion_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / 'PIPAL'
            dest_dir = root / 'Train_dis'
            label_file = root / 'train_labels.txt'

            for folder_name in ['Distortion_1', 'Distortion_2', 'Distortion_3', 'Distortion_4']:
                (source_root / folder_name).mkdir(parents=True, exist_ok=True)

            (source_root / 'Distortion_1' / 'keep_a.bmp').write_bytes(b'data')
            (source_root / 'Distortion_2' / 'keep_b.bmp').write_bytes(b'data')
            (source_root / 'Distortion_4' / 'keep_c.bmp').write_bytes(b'data')
            (source_root / 'Distortion_3' / 'extra.bmp').write_bytes(b'data')

            label_file.write_text('keep_a.bmp\nkeep_b.bmp\nkeep_c.bmp\n')

            prepare_train_dis(source_root, dest_dir, label_file)

            self.assertTrue((dest_dir / 'keep_a.bmp').exists())
            self.assertTrue((dest_dir / 'keep_b.bmp').exists())
            self.assertTrue((dest_dir / 'keep_c.bmp').exists())
            self.assertFalse((dest_dir / 'extra.bmp').exists())
            self.assertEqual(len(list(dest_dir.iterdir())), 3)

    def test_prepare_train_dis_replaces_existing_destination_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / 'PIPAL'
            dest_dir = root / 'Train_dis'
            label_file = root / 'train_labels.txt'

            for folder_name in ['Distortion_1', 'Distortion_2']:
                (source_root / folder_name).mkdir(parents=True, exist_ok=True)

            (source_root / 'Distortion_1' / 'dup.bmp').write_bytes(b'first')
            (source_root / 'Distortion_2' / 'dup.bmp').write_bytes(b'second')
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / 'dup.bmp').write_bytes(b'old')
            label_file.write_text('dup.bmp\n')

            prepare_train_dis(source_root, dest_dir, label_file)

            self.assertTrue((dest_dir / 'dup.bmp').exists())
            self.assertIn((dest_dir / 'dup.bmp').read_bytes(), [b'first', b'second'])


if __name__ == '__main__':
    unittest.main()
