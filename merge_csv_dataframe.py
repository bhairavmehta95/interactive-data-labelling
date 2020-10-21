import h5py
import csv
import numpy as np

import os.path as osp

if __name__ == "__main__":
    env = 'FetchSlide'
    for epoch in range(21, 31, 3):
        for episode in range(50):
            reward_csv = f"reward_{env}_{epoch}_{episode}.csv"
            episode_file = f'{env}-v1.h5'

            if not osp.exists(reward_csv): continue

            rewards = []
            with open(reward_csv, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ')
                for row in csvreader:
                    rewards.append(float(row[1]))

            with h5py.File(episode_file, 'a') as hf:
                annotation_path = f'epoch_{epoch}/{episode}/annotations'
                hf.create_dataset(annotation_path, data=rewards, dtype='f')

            with h5py.File(episode_file, 'r') as hf:
                print(list(hf.keys()))
