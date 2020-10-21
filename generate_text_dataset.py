import h5py
import os.path as osp

import numpy as np

DATASET_TYPES = {
    'expert': {
        'FetchSlide': [27, 30],
        # 'FetchReach': [6],
        # 'FetchPickAndPlace': [24, 27, 30],
        # 'FetchPush': [24, 27, 30],
    },
    'medium': {
        'FetchSlide': [21, 24, 27, 30],
        # 'FetchReach': [6],
        # 'FetchPickAndPlace': [18, 21, 24, 27, 30],
        # 'FetchPush': [18, 21, 24, 27, 30],
    }
}

# FetchReach: obs[:3] + obs[:-3]
# Others: obs[:6] + obs[:-3]


if __name__ == "__main__":
    datatypes = list(DATASET_TYPES.keys()) #+ ['all']
    for env in DATASET_TYPES['expert'].keys():
        episode_file = f'{env}-v1.h5'
        obsindex = 3 if env == 'FetchReach' else 6

        with h5py.File(episode_file, 'r') as hf:
            for datatype in datatypes:
                observations = []
                rewards = []

                for epoch in range(3, 31, 3):
                    if env == 'FetchReach' and epoch > 6: break
                    if env == 'FetchPush' and epoch == 30: break

                    if datatype is not 'all':
                        if epoch not in DATASET_TYPES[datatype][env]: continue

                    for episode in range(10):
                        annotation_path = f'epoch_{epoch}/{episode}/annotations'
                        observation_path = f'epoch_{epoch}/{episode}/observations'
                        achieved_goal_path = f'epoch_{epoch}/{episode}/achieved_goals'

                        try:
                            ep_rew = hf[annotation_path]
                            ep_obs = hf[observation_path]
                            ep_ag = hf[achieved_goal_path]

                            truncated_obs = np.concatenate([ep_ag, ep_obs[:, -3:]], axis=1)
                        except:
                            print(f'Skipping {env}|{episode}|{epoch}') 

                        observations.append(truncated_obs)
                        rewards.append(ep_rew)
                
                observations = np.concatenate(observations)
                rewards = np.concatenate(rewards)

                
                dataset = np.concatenate([observations, np.array(rewards)[:, np.newaxis]], axis=1)
                np.savetxt(f'{env}-v1-{datatype}-annotation-achievedgoal.txt', dataset)
                
