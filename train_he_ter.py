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
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import tenseal as ts  # 导入TenSEAL库用于同态加密

# 初始化 TenSEAL 加密上下文
def initialize_encryption_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        poly_modulus_degree=32768, 
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

# 加密模型参数
def encrypt_model_parameters(model, encryption_context):
    encrypted_params = {}
    for name, param in model.state_dict().items():
        tensor = param.cpu().flatten().numpy()  # 将参数移到CPU并转换为numpy数组
        encrypted_tensor = ts.ckks_vector(encryption_context, tensor)
        encrypted_params[name] = encrypted_tensor
    return encrypted_params

# 聚合加密的模型参数
def aggregate_encrypted_models(encrypted_models, data_sizes):
    total_data = sum(data_sizes)
    aggregated_params = {}

    for name in encrypted_models[0]:
        aggregated_params[name] = encrypted_models[0][name].copy()

    for i in range(1, len(encrypted_models)):
        for name in encrypted_models[i]:
            aggregated_params[name] += encrypted_models[i][name]

    for name in aggregated_params:
        aggregated_params[name] *= 1 / total_data
    
    return aggregated_params

# 解密模型参数
def decrypt_model_parameters(encrypted_params, encryption_context, model):
    decrypted_params = {}
    for name, enc_tensor in encrypted_params.items():
        if isinstance(enc_tensor, list):  # 如果enc_tensor是一个列表，需要对每个元素进行处理
            decrypted_tensor = [ts.ckks_vector(encryption_context, t).decrypt() for t in enc_tensor]
        else:
            decrypted_tensor = enc_tensor.decrypt()

        # 将解密后的tensor重新调整为模型参数的形状
        original_shape = model.state_dict()[name].shape
        decrypted_params[name] = torch.tensor(decrypted_tensor).view(original_shape)
    return decrypted_params

# 计算准确率
def calculate_acc(prob, label):
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1
    return np.array(acc_train)

# 采样概率
def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1] - 1
    label = label.view(-1)
    init_label = np.linspace(0, num_label - 1, num_label)
    init_prob = torch.zeros(size=(num_label, num_neg + len(label)))

    random_ig = random.sample(range(1, l_m + 1), num_neg)
    while len([lab for lab in label if lab in random_ig]) != 0:
        random_ig = random.sample(range(1, l_m + 1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i - len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)

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

# 评估模型
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
        valid_size, test_size = 0, 0
        acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

        for loader in self.loaders:
            bar = tqdm(total=part)
            for step, item in enumerate(loader):
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0] + 1):
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    eval_input = person_input * input_mask
                    eval_m1 = person_m1 * m1_mask
                    eval_m2t = person_m2t[:, mask_len - 1]
                    eval_label = person_label[:, mask_len - 1]
                    eval_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(eval_input, eval_m1, self.mat2s, eval_m2t, eval_len)

                    if mask_len == person_traj_len[0] - 1:
                        valid_size += person_input.shape[0]
                        acc_valid += calculate_acc(prob, eval_label)
                    elif mask_len == person_traj_len[0]:
                        test_size += person_input.shape[0]
                        acc_test += calculate_acc(prob, eval_label)

                bar.update(self.batch_size)
            bar.close()

        acc_valid = np.array(acc_valid) / valid_size
        acc_test = np.array(acc_test) / test_size
        print('Aggregated model - valid_acc:{}'.format(acc_valid))
        print('Aggregated model - test_acc:{}'.format(acc_test))

        return acc_valid, acc_test

# 可视化结果
def plot_metrics(records):
    epochs = records['epoch']
    acc_valid = np.array(records['acc_valid'])
    acc_test = np.array(records['acc_test'])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i, k in enumerate([1, 5, 10, 20]):
        plt.plot(epochs, acc_valid[:, i], label=f'valid_acc@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, k in enumerate([1, 5, 10, 20]):
        plt.plot(epochs, acc_test[:, i], label=f'test_acc@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics_plot20.png')
    plt.show()

if __name__ == '__main__':
    # 初始化加密上下文
    encryption_context = initialize_encryption_context()

    # 加载数据
    dname = 'NYC'
    file_client1 = open('./data/' + dname + '_client1_data.pkl', 'rb')
    file_client2 = open('./data/' + dname + '_client2_data.pkl', 'rb')
    file_data1 = joblib.load(file_client1)
    file_data2 = joblib.load(file_client2)

    # 提取数据
    trajs1, mat1_1, mat2s_1, mat2t_1, labels1, lens1, u_max1, l_max1 = file_data1
    trajs2, mat1_2, mat2s_2, mat2t_2, labels2, lens2, u_max2, l_max2 = file_data2

    # 合并 mat2s (保持一致性)
    assert np.array_equal(mat2s_1, mat2s_2), "mat2s arrays do not match between clients"
    mat2s = torch.FloatTensor(mat2s_1).to(device)

    part = 100

    # 准备数据
    trajs1, mat1_1, mat2t_1, labels1, lens1 = trajs1[:part], mat1_1[:part], mat2t_1[:part], labels1[:part], lens1[:part]
    trajs2, mat1_2, mat2t_2, labels2, lens2 = trajs2[:part], mat1_2[:part], mat2t_2[:part], labels2[:part], lens2[:part]

    # 计算全局最大值和最小值
    global_ex = (
        max(mat1_1[:, :, :, 0].max(), mat1_2[:, :, :, 0].max()),
        min(mat1_1[:, :, :, 0].min(), mat1_2[:, :, :, 0].min()),
        max(mat1_1[:, :, :, 1].max(), mat1_2[:, :, :, 1].max()),
        min(mat1_1[:, :, :, 1].min(), mat1_2[:, :, :, 1].min())
    )

    stan = Model(t_dim=hours + 1, l_dim=max(l_max1, l_max2) + 1, u_dim=max(u_max1, u_max2) + 1, embed_dim=50, ex=global_ex, dropout=0)
    num_params = 0

    for name in stan.state_dict():
        print(name)

    for param in stan.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    if load:
        checkpoint = torch.load('best_stan_win_' + dname + '.pth')
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
        agg_records = {'epoch': [], 'acc_valid': [], 'acc_test': []}  # 记录聚合模型评估结果
        start = time.time()

    num_clients = 2
    user_data = [
        (trajs1, mat1_1, mat2s, mat2t_1, labels1, lens1),
        (trajs2, mat1_2, mat2s, mat2t_2, labels2, lens2)
    ]
    data_sizes = [len(user_data[i][0]) for i in range(num_clients)]  # 获取每个客户端的数据量

    print("Data sizes for each client:", data_sizes)

    global_model = copy.deepcopy(stan)
    for epoch in range(5):  # 进行多轮联邦学习训练
        encrypted_models = []

        # 每个客户端进行本地训练
        for user_id in range(num_clients):
            user_model = copy.deepcopy(global_model)  # 为每个客户端分配初始模型
            trainer = Trainer(user_model, records, user_data[user_id])
            trainer.train()

            # 加密本地模型参数
            encrypted_params = encrypt_model_parameters(trainer.model, encryption_context)
            encrypted_models.append(encrypted_params)

        # 服务器端聚合加密的模型参数
        aggregated_encrypted_params = aggregate_encrypted_models(encrypted_models, data_sizes)

        # 客户端解密聚合后的模型参数并更新
        decrypted_params = decrypt_model_parameters(aggregated_encrypted_params, encryption_context, global_model)
        global_model.load_state_dict(decrypted_params)

        # 评估聚合后的全局模型
        evaluator = AggregatedEvaluator(global_model, [user_data[0]], mat2s)
        acc_valid, acc_test = evaluator.evaluate()
        agg_records['acc_valid'].append(acc_valid)
        agg_records['acc_test'].append(acc_test)
        agg_records['epoch'].append(epoch + 1)

        print(f"After epoch {epoch + 1}: agg_records = {agg_records}")

    # 可视化结果
    plot_metrics(agg_records)
