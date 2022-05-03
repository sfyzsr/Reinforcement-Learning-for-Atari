import numpy as np
import random

def sample_n_unique(sampling_f, n):
    # Sampling n object from sampling_f function
    sampleList = []
    while len(sampleList) < n:
        sample = sampling_f()

        if sample not in sampleList:
            sampleList.append(sample)

    return sampleList

class ReplayBuffer(object):
    def __init__(self, size, frame_len):
        # Sequention uniform sample experience replay buffer
        # Only store each frame once. (the next frame after the n th frame is n+1 th frame)
        # Store frames' dtype as np.uint8: 8-bit unsigned integer (range: 0 through 255) 
        # When store 1 000 000 replay, it will cost ram about 1e6*84*84*8 bits = 6.571 GB 

        self.size = int(size)
        self.frame_len = frame_len

        self.next_idx = 0
        self.buffer_size = 0 

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        # Returns true if batch_size can be sampled from the buffer.
        return batch_size + 1 <= self.buffer_size

    def _encode_sample(self, idxes):

        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        # print(obs_batch.shape)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        # Sample by the indexs, and return the minibatch from the replay
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.buffer_size - 2), batch_size)
        return self._encode_sample(idxes)

    def get_recent_observation(self):
        # Return the most recent frames.
        assert self.buffer_size > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):

        start_idx = idx - self.frame_len + 1 

        # if there weren't enough frames in replay
        if start_idx < 0 and self.buffer_size != self.size:
            start_idx = 0

        # if the previous historical frame is the end of the episode, only the newly started episode frame is taken
        for i in range(start_idx, idx):
            if self.done[i % self.size]:
                start_idx = i + 1

        total_missing = self.frame_len - (idx - start_idx + 1)

        # if zero padding is needed for missing context or idx is on the boundry of the buffer 
        if start_idx < 0 or total_missing > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(total_missing)]

            for i in range(start_idx, idx+1):
                frames.append(self.obs[i % self.size])

            return np.concatenate(frames, 0) # c, h, w instead of h, w c

        else:
            # c, h, w instead of h, w c
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            # print(self.obs[start_idx:idx+1].reshape(-1, img_h, img_w).shape)
            return self.obs[start_idx:idx+1].reshape(-1, img_h, img_w)

    def store_frame(self, frame):
        # Store a single frame in the buffer at the next available index
        # Retrun the current storing frame's idx to call store_action_reward_done() function
        
        # transpose image frame from h, w, c to c, h, w 
        frame = frame.transpose(2, 0, 1)

        if self.obs is None:

            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)

        self.obs[self.next_idx] = frame

        current_idx = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.buffer_size = min(self.size, self.buffer_size + 1)

        return current_idx

    def store_action_reward_done(self, idx, action, reward, done):
        # Store the action, reward and done flag for the store_frame() function
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

# idxes = sample_n_unique(lambda: random.randint(0, 100 - 2), 32)
# print(len(idxes))