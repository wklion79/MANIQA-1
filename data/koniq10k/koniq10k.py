import os
import torch
import numpy as np
import cv2


class Koniq10k(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio,
                 crop_mode="base_random", crop_size=224, score_range=None,
                 horizontal_flip_prob=0.0):
        super(Koniq10k, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.horizontal_flip_prob = horizontal_flip_prob

        score_by_file = {}
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                score_by_file[dis] = float(score)

        # Preserve the seeded split order. Re-reading in label-file order would
        # turn a reduced run into a biased "first N rows" subset.
        dis_files_data = [dis for dis in list_name if dis in score_by_file]
        score_data = [score_by_file[dis] for dis in dis_files_data]

        if 0 < keep_ratio < 1:
            keep_count = max(1, int(len(dis_files_data) * keep_ratio))
            dis_files_data = dis_files_data[:keep_count]
            score_data = score_data[:keep_count]

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data, dtype=np.float32)
        score_data = self.normalization(score_data, score_range)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data, score_range=None):
        if score_range is None:
            score_min, score_max = np.min(data), np.max(data)
        else:
            score_min, score_max = score_range
        score_span = score_max - score_min
        if score_span <= 0:
            raise ValueError("Score normalization range must be greater than zero.")
        return (data - score_min) / score_span

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def _normalize_to_tensor(self, image):
        image = image.astype('float32') / 255
        image = np.transpose(image, (2, 0, 1))
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image).type(torch.FloatTensor)

    def _ensure_min_size(self, image):
        h, w = image.shape[:2]
        if h >= self.crop_size and w >= self.crop_size:
            return image

        scale = self.crop_size / min(h, w)
        new_w = int(np.ceil(w * scale))
        new_h = int(np.ceil(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _fixed_local_crops(self, image):
        image = self._ensure_min_size(image)
        h, w = image.shape[:2]
        crop = self.crop_size
        positions = [
            (0, 0),
            (0, w - crop),
            ((h - crop) // 2, (w - crop) // 2),
            (h - crop, 0),
            (h - crop, w - crop),
        ]
        return [image[top:top + crop, left:left + crop] for top, left in positions]
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        image_path = os.path.join(self.dis_path, d_img_name)
        d_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if d_img is None or d_img.size == 0:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        score = self.data_dict['score_list'][idx]

        if self.crop_mode == "global_fixed5":
            if self.horizontal_flip_prob > 0 and np.random.random() < self.horizontal_flip_prob:
                d_img = np.ascontiguousarray(d_img[:, ::-1])
            global_img = cv2.resize(
                d_img,
                (self.crop_size, self.crop_size),
                interpolation=cv2.INTER_CUBIC
            )
            local_imgs = self._fixed_local_crops(d_img)
            sample = {
                'd_img_global': self._normalize_to_tensor(global_img),
                'd_img_local': torch.stack([self._normalize_to_tensor(img) for img in local_imgs]),
                'score': torch.from_numpy(score).type(torch.FloatTensor),
                'image_name': d_img_name
            }
            return sample

        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        sample['image_name'] = d_img_name
        return sample
