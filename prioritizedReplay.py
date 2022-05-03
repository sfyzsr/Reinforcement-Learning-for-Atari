import numpy as np
import random

data_dtype = np.dtype([('state', np.uint8, (1,84, 84)), ('action', np.int32), ('reward', np.float32), ('done', np.bool_)])
blank_frame = (np.zeros((1,84, 84), dtype=np.uint8), 0, 0.0, False)
class SumTree(object):

    def __init__(self, capacity):
        self.capacity = capacity  # capacity for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.array([blank_frame] * capacity, dtype=data_dtype)
        self.data_pointer = 0   # data pointer, not the priority
        self.full = False

    def add(self, p, data):
    # add data with priority
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
    # if tree is full then reset the pointer
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0
            self.full = True

    def update(self, tree_idx, p):
    # update the priority
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:   
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, pri):
    # Get the leaf by the priority
        parent_idx = 0
        while True:   
            left_idx = 2 * parent_idx + 1        
            right_idx = left_idx + 1

            if left_idx >= len(self.tree):      
                leaf_idx = parent_idx
                break

            else:       
                if pri <= self.tree[left_idx]:
                    parent_idx = left_idx
                else:
                    pri -= self.tree[left_idx]
                    parent_idx = right_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_p(self):
    # get the total priority from the root
        return self.tree[0]  
    
    def getDataFromIdx(self, idx):
        return self.data[idx%self.capacity]


class prioritizedReplay(object):  # stored as ( s, a, r ) in SumTree

    def __init__(self, size, frame_len):
        self.size = size
        self.tree = SumTree(int(size))
        self.frame_len = frame_len

        self.prio_max = 0.1
        self.a = 0.6
        self.beta = 0.4
        self.e = 0.01

        self.buffer_size = 0
        self.beta_increment_per_sampling = 0.001 
        self.next_idx = 0
    
    def store(self, state, action, reward, done):
        
        max_p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority

        self.tree.add(max_p, (state, action, reward, done))   # set the max p for new p

        self.buffer_size = min(self.size, self.buffer_size + 1)
        self.next_idx = (self.next_idx + 1) % self.size

    def can_sample(self, batch_size):
        # Returns true if batch_size can be sampled from the buffer.
        return batch_size + 1 <= self.buffer_size
    
    def get_recent_observation(self):
        # Return the most recent frames.
        assert self.buffer_size > 0
        return self._encode_frames((self.next_idx - 1) % self.size)

    def sample(self, batch_size):

        obs_batch      = np.empty([batch_size] + [4,84,84], dtype=np.uint8)
        action_batch   = np.empty([batch_size],                     dtype=np.int32)
        reward_batch   = np.empty([batch_size],                     dtype=np.float32)
        done_batch    = np.empty([batch_size],                     dtype=np.bool)
        next_obs_batch  = np.empty([batch_size] + [4,84,84], dtype=np.uint8)

        idxs = []
        segment = self.tree.total_p() / batch_size
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data_idx = self.tree.get_leaf(s)

            state, action, reward, done = self.tree.data[data_idx]
            
            obs_batch[i] = self._encode_frames(data_idx)
            action_batch[i] = action
            reward_batch[i] = reward
            done_batch[i] = done
            next_obs_batch[i] = self._encode_frames(data_idx+1)
            priorities.append(p)
            idxs.append(idx)
        
        sampling_probabilities = priorities / self.tree.total_p()
        ISweight = np.power(self.tree.data_pointer * sampling_probabilities, -self.beta)
        ISweight = ISweight/ISweight.max()
        return idxs,obs_batch,action_batch,reward_batch,next_obs_batch,done_batch,ISweight

    def _encode_frames(self, data_idx):
        
        data_idx = int (data_idx)

        start_idx = data_idx-self.frame_len+1
        if start_idx < 0 and self.buffer_size != self.size:
            start_idx = 0

        idxs = np.arange(-self.frame_len + 1,  1) + data_idx
        data = self.tree.getDataFromIdx(idxs)
        state = data ['state']
        done  = data ['done']

        strat_idx2 = 0
        # if the previous historical frame is the end of the episode, only the newly started episode frame is taken
        for i in range(0, 3):
            if done[i]:
                strat_idx2 = i + 1

        total_missing = self.frame_len - (3 - strat_idx2 + 1)

        # if zero padding is needed for missing context or idx is on the boundry of the buffer 
        if start_idx < 0 or total_missing > 0:
            frames = [np.zeros_like(state[0]) for _ in range(total_missing)]

            for i in range(strat_idx2, 3+1):
                frames.append(state[i])

            frames = np.concatenate(frames, 0) # c, h, w instead of h, w c
            # print(frames.shape)
            return frames   # (4,84,84) numpy.ndarry
        else:
            # c, h, w instead of h, w c
            img_h, img_w = state.shape[2], state.shape[3]
            # print(self.obs[start_idx:idx+1].reshape(-1, img_h, img_w).shape)
            frames = state[strat_idx2:3+1].reshape(-1, img_h, img_w)
            # print(frames.shape)
            return frames   # (4,84,84) numpy.ndarry

    def update(self, idxs, errors):
        # update tree priority
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
