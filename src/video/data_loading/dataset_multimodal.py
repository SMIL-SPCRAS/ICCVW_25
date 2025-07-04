# coding: utf-8

import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor

class DatasetVideo(Dataset):
    """
    Датасет для формирования обучающих данных по видео.
    """

    def __init__(
        self,
        csv_path,
        video_dir,
        config,
        split,
        image_feature_extractor,
        dataset_name,
        task
    ):
        """
        :param csv_path: Путь к CSV-файлу
        :param video_dir: Путь к видео.
        :param split: "train", "dev" или "test".
        :param image_feature_extractor: Экстрактор видео признаков
        :param subset_size: Если > 0, используется только первые N дивео из CSV (для отладки).
        :param dataset_name: Название корпуса
        """
        super().__init__()
        self.split = split
        self.video_dir = video_dir
        self.image_feature_extractor = image_feature_extractor
        self.subset_size    = config.subset_size
        self.seed = config.random_seed
        self.dataset_name = dataset_name
        self.emotion_columns = config.emotion_columns
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path = config.save_feature_path
        self.roi_video = config.roi_video # face or scene
        self.counter_need_frames = config.counter_need_frames
        self.image_size = config.image_size
        self.image_model_type = config.image_model_type
        if  self.image_model_type == 'clip':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna() # убераем кадры где не найдено лица
        if self.subset_size > 0:
            self.need_segment_name = list(set(self.df.video_name.tolist()))[:self.subset_size]
            self.df = self.df[self.df.video_name.isin(self.need_segment_name)]
            logging.info(f"[DatasetVideo] Используем только {len(self.df)} записей (subset_size={self.subset_size}).")
        else:
            self.need_segment_name = list(set(self.df.video_name.tolist()))

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Ошибка: директория с видео {self.video_dir} не существует!")

        if self.save_prepared_data:
            self.meta = []
            meta_filename = 'task_{}_{}_{}_seed_{}_subset_size_{}_seg_{}_roi_{}_video_model_{}_feature_norm_{}_pred.pickle'.format(
                task,
                self.dataset_name,
                self.split,
                self.seed,
                self.subset_size,
                self.counter_need_frames,
                self.roi_video,
                self.image_model_type,
                config.emb_normalize,
            )

            pickle_path = os.path.join(self.save_feature_path, meta_filename)
            self.load_data(pickle_path)

            if not self.meta:
                self.prepare_data()
                os.makedirs(self.save_feature_path, exist_ok=True)
                self.save_data(pickle_path)

    def save_data(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as handle:
                self.meta = pickle.load(handle)
        else:
            self.meta = []

    def __len__(self):
        if self.save_prepared_data:
            return len(self.meta)
        else:
            return len(self.need_segment_name)

    def select_uniform_frames(self, frames, N):
        if len(frames) <= N:
            return frames  # Если кадров меньше N, вернуть все
        else:
            np.random.seed(self.seed)
            indices = np.linspace(0, len(frames) - 1, num=N, dtype=int)
            return [frames[i] for i in indices]

    def pth_processing(self, fp):
        class PreprocessInput(torch.nn.Module):
            def init(self):
                super(PreprocessInput, self).init()

            def forward(self, x):
                x = x.to(torch.float32)
                x = torch.flip(x, dims=(0,))
                x[0, :, :] -= 91.4953
                x[1, :, :] -= 103.8827
                x[2, :, :] -= 131.0912
                return x

        def get_img_torch(img):
            if self.image_model_type == "emoresnet50" or self.image_model_type == "emo":
                ttransform = transforms.Compose(
                    [transforms.PILToTensor(), PreprocessInput()]
                )
            elif self.image_model_type == 'resnet18':
                ttransform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

            if self.image_model_type != 'clip':
                img = img.resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
                img = ttransform(img)
                img = torch.unsqueeze(img, 0).to("cuda")
            if self.image_model_type == 'clip':
                img = self.processor(images=img, return_tensors="pt").to("cuda")
                img = img['pixel_values']
            return img

        return get_img_torch(fp)
    
    def draw_box(self, image, box, color=(255, 0, 255)):
        """Draw a rectangle on the image."""
        line_width = 2
        lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    def get_data(self, segment_name):

        curr_data = self.df[self.df.video_name==segment_name] # отбираем все строки нужного сегмента видео
        curr_data = curr_data.dropna() # убераем кадры где не найдено лица

        # label_vec = np.zeros(self.len_labels) # делаем метки
        # label_vec[int(curr_data.diagnosis.unique()[0])] += 1

        label_vec = curr_data[self.emotion_columns].values[0]
        curr_frames = list(set(curr_data.frame.tolist())) # считываем фреймы
        need_curr_frames = self.select_uniform_frames(curr_frames, self.counter_need_frames)

        full_path_video = os.path.join(self.video_dir, curr_data.video_name.unique()[0])

        # bbox = curr_data[["startX_face","startY_face","endX_face","endY_face"]].value()

        cap = cv2.VideoCapture(full_path_video)
        counter = 1

        all_frames = []
        frame_to_faces_indices = {}

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            if counter in need_curr_frames:
                idx = need_curr_frames.index(counter)
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                if self.roi_video == 'face':
                    bboxs = curr_data[curr_data.frame==counter][["startX_face","startY_face","endX_face","endY_face"]].values.astype('int')
                    frame_to_faces_indices[idx] = range(idx, idx + len(bboxs))
                    for bbox in bboxs:
                        curr_fr = im0[bbox[1]: bbox[3], bbox[0]: bbox[2]]
                        curr_fr = self.pth_processing(Image.fromarray(curr_fr))
                        all_frames.append(curr_fr)

                        # self.draw_box(im0, [bbox[0], bbox[1], bbox[2], bbox[3]])
                else:
                    curr_fr = im0
                    curr_fr = self.pth_processing(Image.fromarray(curr_fr))
                    all_frames.append(curr_fr)
            counter += 1

            # plt.imshow(im0)
            # plt.show()
        cap.release()
        
        if self.roi_video == 'face':
            all_frames = torch.cat(all_frames, dim=0)
            video_features = self.image_feature_extractor.extract(all_frames).to('cpu')

            frame_features = []
            for idx in range(len(need_curr_frames)):
                if idx in frame_to_faces_indices:
                    face_indices = frame_to_faces_indices[idx]
                    if not face_indices:
                        # print(0) 
                        frame_features.append(torch.zeros(video_features.shape[1]))
                    else:
                        # print(1, face_indices)
                        frame_features.append(video_features[list(face_indices)].mean(dim=0))
            video_features = torch.stack(frame_features)
        else:
            all_frames = torch.cat(all_frames, dim=0)
            video_features = self.image_feature_extractor.extract(all_frames).to('cpu')


        torch.cuda.empty_cache()
        

        return {
            "audio_path": full_path_video,
            "audio": 1,
            "video": video_features,
            "label": torch.tensor(label_vec, dtype=torch.float32),
            "text": '',
            "audio_pred": 1,
            "text_pred": 1
        }

    def prepare_data(self):
        """
        Загружает и обрабатывает один элемент датасета (он‑the‑fly).
        """

        for idx, segment_name in enumerate(tqdm(self.need_segment_name)):
            curr_dict = self.get_data(segment_name)

            self.meta.append(
            curr_dict
            )

    def __getitem__(self, index):
        if self.save_prepared_data:
            return self.meta[index]
        else:
            return self.get_data(self.need_segment_name[index])
        
class DatasetMultimodal(Dataset):
    """
    Датасет для формирования обучающих мультимодальных данных.
    """

    def __init__(
        self,
        csv_path,
        config,
        split,
        dataset_name,
        task,
        segment
    ):
        """
        :param csv_path: Путь к CSV-файлу.
        :param split: "train", "dev" или "test".
        :param dataset_name: Название корпуса
        """
        super().__init__()
        self.split = split
        self.segment = segment
        self.subset_size = config.subset_size
        self.seed = config.random_seed
        self.dataset_name = dataset_name
        self.emotion_columns = config.emotion_columns
        self.compound_emotion_columns = ["fearfully surprised","happily surprised","sadly surprised","disgustedly surprised","angrily surprised","sadly fearful","sadly angry"]
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path = config.save_feature_path

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")
        self.df = pd.read_csv(csv_path)
        # self.df = self.df.drop('text', axis=1).dropna() # убераем видео для которых нет лейблов
        self.df = self.df.rename(columns={
            "FS_Anger": "FS_Happiness",
            "FS_Disgust": "FS_Sadness",
            "FS_Fear": "FS_Surprise",
            "FS_Happiness": "FS_Fear",
            "FS_Sadness": "FS_Disgust",
            "FS_Surprise": "FS_Anger"
        })

        # {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
        if self.subset_size > 0:
            self.need_segment_name = list(set(self.df.video_name.tolist()))[:self.subset_size]
            self.df = self.df[self.df.video_name.isin(self.need_segment_name)]
            logging.info(f"[DatasetVideo] Используем только {len(self.df)} записей (subset_size={self.subset_size}).")
        else:
            self.need_segment_name = list(set(self.df.video_name.tolist()))

        if self.save_prepared_data:
            self.meta = []
            meta_filename = 'task_{}_{}_{}_{}_seed_{}_subset_size_{}.pickle'.format(
                task,
                self.dataset_name,
                self.segment,
                self.split,
                self.seed,
                self.subset_size,
            )

            pickle_path = os.path.join(self.save_feature_path, meta_filename)
            self.load_data(pickle_path)

            if not self.meta:
                self.prepare_data()
                os.makedirs(self.save_feature_path, exist_ok=True)
                self.save_data(pickle_path)

    def save_data(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as handle:
                self.meta = pickle.load(handle)
        else:
            self.meta = []

    def __len__(self):
        if self.save_prepared_data:
            return len(self.meta)
        else:
            return len(self.need_segment_name)

    def get_data(self, segment_name):

        curr_data = self.df[self.df.video_name==segment_name] # отбираем все строки нужного сегмента видео

        label_vec = curr_data[self.emotion_columns].values[0]

        audio = curr_data[['A_'+ i for i in self.emotion_columns]].values[0]
        static_face = curr_data[['FS_'+ i for i in self.emotion_columns[:-1]]].values[0]
        dynamic_face = curr_data[['FD_'+ i for i in self.emotion_columns]].values[0]
        text = curr_data[['T_'+ i for i in self.emotion_columns]].values[0]
        scene = curr_data[['S_'+ i for i in self.emotion_columns]].values[0]
        scene_clip = curr_data[['C_'+ i for i in self.emotion_columns]].values[0]
        scene_logit = curr_data[['NSL_'+ i for i in self.emotion_columns]].values[0]
        multimodal_logit = curr_data[['MML_'+ i for i in self.emotion_columns]].values[0]
        scene_prob = curr_data[['NSP_'+ i for i in self.emotion_columns]].values[0]
        multimodal_prob = curr_data[['MMP_'+ i for i in self.emotion_columns]].values[0]       
        scene_clip = curr_data[['C_'+ i for i in self.emotion_columns]].values[0]
        compound_clip = curr_data[self.compound_emotion_columns].values[0]

        return {
            "video_path": segment_name,
            "audio": torch.tensor(audio, dtype=torch.float32),
            "static_face": torch.tensor(static_face.tolist()+[0.0], dtype=torch.float32),
            "dynamic_face": torch.tensor(dynamic_face, dtype=torch.float32),
            "text": torch.tensor(text, dtype=torch.float32),
            "scene": torch.tensor(scene, dtype=torch.float32),
            "scene_clip": torch.tensor(scene_clip, dtype=torch.float32),
            "scene_prob": torch.tensor(scene_prob, dtype=torch.float32),
            "scene_logit": torch.tensor(scene_logit, dtype=torch.float32),
            "multimodal_logit": torch.tensor(multimodal_logit, dtype=torch.float32),
            "multimodal_prob": torch.tensor(multimodal_prob, dtype=torch.float32),
            "compound_clip": torch.tensor(compound_clip, dtype=torch.float32),            
            "label": torch.tensor(label_vec, dtype=torch.float32),
        }

    def prepare_data(self):
        """
        Загружает и обрабатывает один элемент датасета (он‑the‑fly).
        """

        for idx, segment_name in enumerate(tqdm(self.need_segment_name)):
            curr_dict = self.get_data(segment_name)

            self.meta.append(
            curr_dict
            )

    def __getitem__(self, index):
        if self.save_prepared_data:
            return self.meta[index]
        else:
            return self.get_data(self.need_segment_name[index])