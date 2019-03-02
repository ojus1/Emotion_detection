from torch.utils.data import Dataset
import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image
import librosa
import os
import numpy as np

def one_hot_encode(x, size):
    temp = [0] * size
    temp[x] = 1
    return temp

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter
from videotransforms.volume_transforms import ClipToTensor

frame_size = (200, 200)
scale_size = (200, 356)


video_transform_list = [
    RandomRotation(20),
    Resize(scale_size),
    RandomCrop(frame_size),
    ColorJitter(0.1, 0.1, 0.1, 0.1),
    ClipToTensor(channel_nb=3)
]

video_transform = Compose(video_transform_list)

# Dataset for the 3D CNN model, returns sequence of frames (every second frame) from video
class VideoFramesDataset(Dataset):
    def __init__(self):
        super(VideoFramesDataset, self).__init__()

        self.root_dir = "../Datasets/emotion_detection/full/"
        self.video_names = os.listdir(self.root_dir)

        '''
        Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        Vocal channel (01 = speech, 02 = song).
        Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        Repetition (01 = 1st repetition, 02 = 2nd repetition).
        Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
        '''
        self.emotion_label_dict = {"01": 0,
                                    "02": 1,
                                    "03": 2,
                                    "04": 3,
                                    "05": 4,
                                    "06": 5,
                                    "07": 6,
                                    "08": 7,}

        self.emotion_intensity_dict = {"01": 0,
                                    "02": 1,}

    def __getitem__(self, idx):
        tags = self.video_names[idx].split("-")
        #one hot encoding the output
        emotion = one_hot_encode(self.emotion_label_dict[tags[2]], 8) 
        intensity = self.emotion_intensity_dict[tags[3]]
        y = np.array(emotion + intensity)

        cap = cv.VideoCapture(self.root_dir + self.video_names[idx])

        frames = list()
        counter = 0
        frames_appended = 0
        while True:
            return_flag, frame = cap.read()
            counter = counter + 1
            
            if not return_flag or frames_appended >= 20:
                break

            if counter % 3 == 0 and frames_appended <= 20:
                frames_appended += 1
                frames.append(Image.fromarray(frame*255))

        frames = video_transform(frames)
        return frames, y

    def __len__(self):
        return len(self.video_names)


if __name__ == "__main__":
    videoDataset = VideoFramesDataset()
    print(videoDataset[0][0].shape)
    print(len(videoDataset))