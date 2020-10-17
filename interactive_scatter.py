import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import matplotlib.cm as cm

import math
import numpy as np
from itertools import islice
import os.path as osp

sns.set()



def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def interpolate(points, dataset_len):
    first_x, first_y = next(iter(points))
    if first_x != 0:
        points.append((0, 0))

    last_x = points[-1][0]
    last_y = points[-1][1]

    if last_x != dataset_len:
        points.append((dataset_len, last_y))

    points = sorted(points)
    endpoints = window(points)

    rewards = np.zeros(dataset_len)

    for pt1, pt2 in endpoints:
        slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        b = pt1[1] - slope * pt1[0]

        for x in range(pt1[0], pt2[0]):
            x = np.clip(x, 0, dataset_len-1)
            rewards[x] = slope * x + b 

    return rewards

class RewardSketcher(object):
    u""" An example of plot with draggable markers """

    def __init__(self, episode_file, env, epoch, episode):
        self._figure, self._axes, self._line = None, None, None
        self._dragging_point = None
        self._points = {}
        self.env = env
        self.epoch = epoch
        self.episode = episode

        if episode_file is not None:
            import h5py
            from skimage.transform import rescale, resize
            from skimage.color import rgb2gray

            with h5py.File(episode_file, 'r') as hf:
                image_path = f'epoch_{epoch}/{episode}/images'
                success_path = f'epoch_{epoch}/{episode}/successes'
                reward_path = f'epoch_{epoch}/{episode}/rewards'
                
                demo_obs = hf[image_path]
                successes = hf[success_path]
                rewards = hf[reward_path]

                self.adjusted_rewards = (np.array(rewards) + 1) * 100
                self.dataset_len = len(demo_obs)
                self.image_arr = np.zeros((self.dataset_len, demo_obs[0].shape[0] // 8, demo_obs[0].shape[1] // 8))

                self._initial_success = self.dataset_len
                for i in range(self.dataset_len):
                    image_resized = rgb2gray(resize(demo_obs[i], (demo_obs[i].shape[0] // 8, demo_obs[i].shape[1] // 8)))
                    self.image_arr[i] = image_resized
                    if successes[i] and self._initial_success == self.dataset_len:
                        self._initial_success = i

        self._init_plot()

    def _init_plot(self):
        self._figure = plt.figure(f"Reward Sketching, {self.env} ({self.epoch}|{self.episode}) ")

        axes = plt.subplot(1, 1, 1)
        axes.set_xlim(0, 50)
        axes.set_ylim(0, 100)
        axes.grid(which="both")
        axes.set_xlabel('frame $\mathcal{I}_t$')
        axes.set_ylabel('reward $r_t$')

        axes.axhline(y=90, xmin=0, xmax=self.dataset_len, c='green')
        axes.axvline(x=self._initial_success, ymin=0, ymax=100)
        axes.plot(np.arange(self.adjusted_rewards.shape[0]), self.adjusted_rewards, "g", marker="*", markersize=7)

        self._axes = axes

        c1 = self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        c2 = self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        c3 = self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        c4 = self._figure.canvas.mpl_connect('key_press_event', self._on_key_press)

        self.callbacks = [c1, c2, c3, c4]
        
        # this is another inset axes over the main axes
        self._image_ax = plt.axes([0.2, 0.6, .2, .2], facecolor='y')
        self._image_ax.set_xticks([])
        self._image_ax.set_yticks([])

        plt.show()

    def _load_image(self, x_coord):
        index = np.clip(int(x_coord), 0, self.dataset_len-1)
        self._image_ax.imshow(self.image_arr[index], cmap=cm.gray)
        self._update_plot()

    def _update_plot(self):
        if not self._points and not self._line:
            self._line, = self._axes.plot([], [], "b", marker="o", markersize=10)
        
        if not self._points:
            self._line.set_data([], []) 
        else:
            x, y = zip(*sorted(self._points.items()))
            # Add new plot
            if not self._line:
                self._line, = self._axes.plot(x, y, "b", marker="o", markersize=10)
            # Update current plot
            else:
                self._line.set_data(x, y)
            
        self._figure.canvas.draw()

    def _add_point(self, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = int(x.xdata), int(x.ydata)
        self._points[x] = y
        return x, y

    def _remove_point(self, x, _):
        if x in self._points:
            self._points.pop(x)

    def _find_neighbor_point(self, event):
        u""" Find point around mouse position

        :rtype: ((int, int)|None)
        :return: (x, y) if there are any point around mouse else None
        """
        distance_threshold = 3.0
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for x, y in self._points.items():
            distance = math.hypot(event.xdata - x, event.ydata - y)
            if distance < min_distance:
                min_distance = distance
                nearest_point = (x, y)
        if min_distance < distance_threshold:
            return nearest_point
        return None

    def _on_click(self, event):
        u""" callback method for mouse click event

        :type event: MouseEvent
        """
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            else:
                self._add_point(event)
            self._update_plot()
        # right click
        elif event.button == 3 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._remove_point(*point)
                self._update_plot()

    def _on_release(self, event):
        u""" callback method for mouse release event

        :type event: MouseEvent
        """
        if event.button == 1 and event.inaxes in [self._axes] and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        u""" callback method for mouse motion event

        :type event: MouseEvent
        """
        if event.xdata is None or event.ydata is None:
            return

        if self._dragging_point:
            self._remove_point(*self._dragging_point)
            self._dragging_point = self._add_point(event)
            self._update_plot()

        self._load_image(event.xdata)

    def _on_key_press(self, event):
        if event.key == 'w':
            reward_csv = f"reward_{self.env}_{self.epoch}_{self.episode}.csv"

            with open(reward_csv, "w") as f:
                points = interpolate(sorted(self._points.items()), self.dataset_len)
                
                for x, y in enumerate(points):
                    adjusted_reward = (y / 100) - 1
                    f.write(f'{x} {adjusted_reward}\n')

            [self._figure.canvas.mpl_disconnect(c) for c in self.callbacks]
            plt.close()

if __name__ == "__main__":
    env = 'FetchSlide'
    for epoch in range(3, 31, 3):
        for episode in range(10):
            reward_csv = f"reward_{env}_{epoch}_{episode}.csv"
            episode_file = f'{env}-v1.h5'
            if not osp.exists(episode_file) or osp.exists(reward_csv): continue
            plot = RewardSketcher(episode_file=episode_file, env=env, epoch=epoch, episode=episode)