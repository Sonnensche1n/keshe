import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from load import max_len
import torch.nn.functional as F
import random

device = torch.device('cpu') # Force CPU since cuda fails in this environment

def to_npy(x):
    return x.cpu().data.numpy() if torch.cuda.is_available() else x.detach().numpy()

def calculate_acc(prob, label):
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1
    return np.array(acc_train)

def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1
    label = label.view(-1)
    init_label = np.linspace(0, num_label-1, num_label)
    init_prob = torch.zeros(size=(num_label, num_neg+len([label])))

    random_ig = random.sample(range(1, l_m+1), num_neg)
    while len([lab for lab in label if lab in random_ig]) != 0:
        random_ig = random.sample(range(1, l_m+1), num_neg)

    global device
    for k in range(num_label):
        for i in range(num_neg + 1):
            if i < num_neg:
                init_prob[k, i] = prob[k, random_ig[i]]
            else:
                init_prob[k, i] = prob[k, label[k]]

    return torch.FloatTensor(init_prob).to(device), torch.LongTensor(init_label).to(device)

class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        self.traj = torch.tensor(traj, dtype=torch.long).to(device)
        self.mat1 = torch.tensor(m1, dtype=torch.float).to(device)
        self.vec = torch.tensor(v, dtype=torch.float).to(device)
        self.label = torch.tensor(label, dtype=torch.long).to(device)
        self.length = torch.tensor(length, dtype=torch.long).to(device)

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):
        return len(self.traj)

class AggregatedEvaluator:
    def __init__(self, model, user_data_list, mat2s):
        self.model = model.to(device)
        self.user_data_list = user_data_list
        self.batch_size = 10
        self.mat2s = mat2s
        self.loaders = []
        for user_data in user_data_list:
            traj, mat1, _, mat2t, label, length = user_data
            dataset = DataSet(traj, mat1, mat2t, label - 1, length)
            loader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
            self.loaders.append(loader)

    def evaluate(self):
        self.model.eval()
        valid_size, test_size = 0, 0
        acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

        with torch.no_grad():
            for loader in self.loaders:
                for step, item in enumerate(loader):
                    person_input, person_m1, person_m2t, person_label, person_traj_len = item

                    input_mask = torch.zeros((person_input.shape[0], max_len, 3), dtype=torch.long).to(device)
                    m1_mask = torch.zeros((person_input.shape[0], max_len, max_len, 2), dtype=torch.float32).to(device)
                    for mask_len in range(1, person_traj_len[0] + 1):
                        input_mask[:, :mask_len] = 1.
                        m1_mask[:, :mask_len, :mask_len] = 1.

                        eval_input = person_input * input_mask
                        eval_m1 = person_m1 * m1_mask
                        eval_m2t = person_m2t[:, mask_len - 1]
                        eval_label = person_label[:, mask_len - 1]
                        eval_len = torch.zeros(size=(person_input.shape[0],), dtype=torch.long).to(device) + mask_len

                        prob = self.model(eval_input, eval_m1, self.mat2s, eval_m2t, eval_len)

                        if mask_len == person_traj_len[0] - 1:
                            valid_size += person_input.shape[0]
                            acc_valid += calculate_acc(prob, eval_label)
                        elif mask_len == person_traj_len[0]:
                            test_size += person_input.shape[0]
                            acc_test += calculate_acc(prob, eval_label)

        acc_valid = np.array(acc_valid) / valid_size if valid_size > 0 else np.zeros(4)
        acc_test = np.array(acc_test) / test_size if test_size > 0 else np.zeros(4)
        
        return acc_valid, acc_test
