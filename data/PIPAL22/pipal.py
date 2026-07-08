import os
import torch
import numpy as np
import cv2


class PIPAL(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, transform, keep_ratio):
        super(PIPAL, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        name_type = {}
        
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                
                # obtain the spliting parts
                name = dis[:-4]
                split_list = dis.split('_')
                img_name, dis_type, level = split_list[0], split_list[1], split_list[2]

                if img_name + '_' + dis_type not in name_type.keys():
                    name_type[img_name + '_' + dis_type] = 1
                else:
                    name_type[img_name + '_' + dis_type] += 1

        count_name_type = {}
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]

                name = dis[:-4]
                split_list = dis.split('_')
                img_name, dis_type, level = split_list[0], split_list[1], split_list[2]

                if img_name + '_' + dis_type not in count_name_type.keys():
                    count_name_type[img_name + '_' + dis_type] = 1
                else:
                    count_name_type[img_name + '_' + dis_type] += 1

                if count_name_type[img_name + '_' + dis_type] <= int(name_type[img_name + '_' + dis_type] * keep_ratio):
                    score = float(score)
                    dis_files_data.append(dis)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        data = np.asarray(data, dtype=np.float32)
        if data.size == 0:
            return data
        data_range = np.max(data) - np.min(data)
        if np.isclose(data_range, 0):
            return np.zeros_like(data, dtype=np.float32)
        return (data - np.min(data)) / data_range

    def _resolve_image_path(self, d_img_name):
        if not d_img_name:
            return None

        normalized_name = os.path.normpath(d_img_name)
        candidate_paths = []
        if os.path.isabs(normalized_name):
            candidate_paths.append(normalized_name)
        else:
            candidate_paths.extend([
                os.path.join(self.dis_path, normalized_name),
                os.path.join(self.dis_path, os.path.basename(normalized_name)),
            ])

        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                return candidate

        for root, _, files in os.walk(self.dis_path):
            if os.path.basename(normalized_name) in files:
                return os.path.join(root, os.path.basename(normalized_name))

        return None

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_path = self._resolve_image_path(d_img_name)
        if d_img_path is None:
            raise FileNotFoundError(f'Could not find image {d_img_name} under {self.dis_path}')

        d_img = cv2.imread(d_img_path, cv2.IMREAD_COLOR)
        if d_img is None:
            raise ValueError(f'Could not read image {d_img_path}')

        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        
        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
