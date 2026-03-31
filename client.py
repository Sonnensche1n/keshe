import torch
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import copy
from utils import DataSet, calculate_acc, sampling_prob
from load import device, max_len
import numpy as np

class STANClient:
    def __init__(self, client_id, user_data, model):
        """
        user_data: (trajs, mat1, mat2s, mat2t, labels, lens)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = user_data
        
        # Prepare dataset
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label - 1, self.len)
        self.batch_size = 10
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        
        self.num_neg = 10
        self.learning_rate = 3e-3

    def local_train(self, global_state_dict, epochs=1):
        # Synchronize with global model
        self.model.load_state_dict(global_state_dict)
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        
        for e in range(epochs):
            for step, item in enumerate(self.data_loader):
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # Masking logic as in STAN
                input_mask = torch.zeros((person_input.shape[0], max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((person_input.shape[0], max_len, max_len, 2), dtype=torch.float32).to(device)
                
                for mask_len_val in range(1, person_traj_len[0] + 1):
                    input_mask[:, :mask_len_val] = 1.
                    m1_mask[:, :mask_len_val, :mask_len_val] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len_val - 1]
                    train_label = person_label[:, mask_len_val - 1]
                    train_len = torch.zeros(size=(person_input.shape[0],), dtype=torch.long).to(device) + mask_len_val

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)

                    if mask_len_val <= person_traj_len[0] - 2:
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        
                        optimizer.zero_grad()
                        loss_train.backward()
                        optimizer.step()

        # Return the updated weights and the number of data samples
        return copy.deepcopy(self.model.state_dict()), len(self.dataset)
