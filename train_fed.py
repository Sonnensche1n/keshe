# 导入库
from load import *
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 计算准确率
def calculate_acc(prob, label):
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1
    return np.array(acc_train)

# 数据集类
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

# 训练器类
class Trainer:
    def __init__(self, model, record, user_data):
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 10
        self.learning_rate = 3e-3
        self.num_epoch = 1
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0

        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = user_data
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label - 1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0] + 1):
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)

                    if mask_len <= person_traj_len[0] - 2:
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    elif mask_len == person_traj_len[0] - 1:
                        valid_size += person_input.shape[0]
                        acc_valid += calculate_acc(prob, train_label)
                    elif mask_len == person_traj_len[0]:
                        test_size += person_input.shape[0]
                        acc_test += calculate_acc(prob, train_label)

                bar.update(self.batch_size)
            bar.close()

            acc_valid = np.array(acc_valid) / valid_size
            print('epoch:{}, time:{}, valid_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_valid))

            acc_test = np.array(acc_test) / test_size
            print('epoch:{}, time:{}, test_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_test))

# 聚合模型
def aggregate_models(models, data_sizes):
    total_data = sum(data_sizes)
    global_model = copy.deepcopy(models[0])
    for key in global_model.state_dict().keys():
        weighted_sum = torch.zeros_like(global_model.state_dict()[key])
        for model, size in zip(models, data_sizes):
            weighted_sum += model.state_dict()[key] * (size / total_data)
        global_model.state_dict()[key] = weighted_sum
    return global_model

# 获取模型梯度向量
def get_model_gradients_vector(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))  # 展开梯度
    return torch.cat(grads).cpu().numpy()  # 转换为numpy数组

# 聚类客户端
def cluster_clients(models, num_clusters):
    gradients = [get_model_gradients_vector(model) for model in models]
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(gradients)
    return labels

# 按比例从每个类中采样客户端
def sample_clients_by_cluster(labels, models, data_sizes, sample_ratio):
    selected_models = []
    selected_sizes = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        sample_count = int(len(cluster_indices) * sample_ratio)
        sampled_indices = random.sample(cluster_indices, sample_count)
        
        for idx in sampled_indices:
            selected_models.append(models[idx])
            selected_sizes.append(data_sizes[idx])
    
    return selected_models, selected_sizes

# 主程序
if __name__ == '__main__':
    dname = 'NYC'
    file_client1 = open('./data/' + dname + '_client1_data.pkl', 'rb')
    file_client2 = open('./data/' + dname + '_client2_data.pkl', 'rb')
    file_data1 = joblib.load(file_client1)
    file_data2 = joblib.load(file_client2)

    trajs1, mat1_1, mat2s_1, mat2t_1, labels1, lens1, u_max1, l_max1 = file_data1
    trajs2, mat1_2, mat2s_2, mat2t_2, labels2, lens2, u_max2, l_max2 = file_data2

    assert np.array_equal(mat2s_1, mat2s_2), "mat2s arrays do not match between clients"
    mat2s = torch.FloatTensor(mat2s_1).to(device)

    part = 100
    trajs1, mat1_1, mat2t_1, labels1, lens1 = trajs1[:part], mat1_1[:part], mat2t_1[:part], labels1[:part], lens1[:part]
    trajs2, mat1_2, mat2t_2, labels2, lens2 = trajs2[:part], mat1_2[:part], mat2t_2[:part], labels2[:part], lens2[:part]

    global_ex = (
        max(mat1_1[:, :, :, 0].max(), mat1_2[:, :, :, 0].max()),
        min(mat1_1[:, :, :, 0].min(), mat1_2[:, :, :, 0].min()),
        max(mat1_1[:, :, :, 1].max(), mat1_2[:, :, :, 1].max()),
        min(mat1_1[:, :, :, 1].min(), mat1_2[:, :, :, 1].min())
    )

    stan = Model(t_dim=hours + 1, l_dim=max(l_max1, l_max2) + 1, u_dim=max(u_max1, u_max2) + 1, embed_dim=50, ex=global_ex, dropout=0)
    
    load = False
    records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
    agg_records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
    start = time.time()

    num_clients = 2
    user_data = [
        (trajs1, mat1_1, mat2s, mat2t_1, labels1, lens1),
        (trajs2, mat1_2, mat2s, mat2t_2, labels2, lens2)
    ]
    data_sizes = [len(user_data[i][0]) for i in range(num_clients)]

    global_model = copy.deepcopy(stan)
    for epoch in range(6):  
        user_models = []
        for user_id in range(num_clients):
            user_model = copy.deepcopy(global_model)
            trainer = Trainer(user_model, records, user_data[user_id])
            trainer.train()
            user_models.append(trainer.model)

        labels = cluster_clients(user_models, num_clusters=2)  
        sampled_models, sampled_sizes = sample_clients_by_cluster(labels, user_models, data_sizes, sample_ratio=0.5)

        global_model = aggregate_models(sampled_models, sampled_sizes)  

        print(f'Finished epoch {epoch + 1}')
        
        evaluator = AggregatedEvaluator(global_model, [user_data[0]], mat2s)
        acc_valid, acc_test = evaluator.evaluate()
        agg_records['acc_valid'].append(acc_valid)
        agg_records['acc_test'].append(acc_test)
        agg_records['epoch'].append(epoch + 1)

    plot_metrics(agg_records)
