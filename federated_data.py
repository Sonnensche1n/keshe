import numpy as np
import torch
from math import radians, cos, sin, asin, sqrt
import joblib
import os
from torch.nn.utils.rnn import pad_sequence

max_len = 100

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def rst_mat1(traj, poi):
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1] - 1], poi[term[1] - 1]
            mat[i, j, 0] = haversine(lon1=poi_item[2], lat1=poi_item[1], lon2=poi_term[2], lat2=poi_term[1])
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat

def rs_mat2s(poi, l_max):
    candidate_loc = np.linspace(1, l_max, l_max)
    mat = np.zeros((l_max, l_max))
    for i, loc1 in enumerate(candidate_loc):
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1) - 1], poi[int(loc2) - 1]
            mat[i, j] = haversine(lon1=poi1[2], lat1=poi1[1], lon2=poi2[2], lat2=poi2[1])
    return mat

def rt_mat2t(traj_time):
    if len(traj_time) < 2:
        return np.zeros((0, 0))
    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time):
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):
            mat[i-1, j] = np.abs(item - term)
    return mat

def process_federated_data(data_file, poi_file, dname, num_clients=2):
    print(f"Loading {data_file}...")
    data = np.load(data_file)
    data[:, -1] = np.array(data[:, -1]/60, dtype=int)
    poi = np.load(poi_file)
    
    num_user = int(data[-1, 0])
    data_user = data[:, 0]
    u_max, l_max = int(np.max(data[:, 0])), int(np.max(data[:, 1]))
    
    print("Computing global spatial matrix (mat2s)...")
    mat2s = rs_mat2s(poi, l_max)
    
    # Identify unique users and shuffle them for non-IID splitting
    all_users = np.unique(data_user)
    all_users = all_users[all_users > 0] # Remove user 0 if exists
    np.random.shuffle(all_users)
    
    # Split users into clients
    client_users = np.array_split(all_users, num_clients)
    
    for client_id in range(num_clients):
        print(f"Processing Client {client_id+1}/{num_clients} with {len(client_users[client_id])} users...")
        trajs, labels, mat1, mat2t, lens = [], [], [], [], []
        
        for u_id in client_users[client_id]:
            user_traj = data[np.where(data_user == u_id)]
            user_traj = user_traj[np.argsort(user_traj[:, 2])].copy()
            
            if len(user_traj) > max_len + 1:
                user_traj = user_traj[-max_len-1:]
                
            user_len = len(user_traj[:-1])
            if user_len < 2:
                continue
                
            user_mat1 = rst_mat1(user_traj[:-1], poi)
            user_mat2t = rt_mat2t(user_traj[:, 2])
            
            init_mat1 = np.zeros((max_len, max_len, 2))
            init_mat2t = np.zeros((max_len, max_len))
            init_mat1[0:user_len, 0:user_len] = user_mat1
            init_mat2t[0:user_len, 0:user_len] = user_mat2t

            trajs.append(torch.LongTensor(user_traj)[:-1])
            mat1.append(init_mat1)
            mat2t.append(init_mat2t)
            labels.append(torch.LongTensor(user_traj[1:, 1]))
            lens.append(user_len-2)

        # Sort by length for RNN padding
        if len(trajs) == 0:
            print(f"Client {client_id+1} has no valid trajectories.")
            continue
            
        zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens), key=lambda x: len(x[0]), reverse=True))
        trajs, mat1, mat2t, labels, lens = zipped
        trajs, mat1, mat2t, labels, lens = list(trajs), list(mat1), list(mat2t), list(labels), list(lens)
        
        trajs = pad_sequence(trajs, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        client_data = [trajs, np.array(mat1), mat2s, np.array(mat2t), labels, np.array(lens), u_max, l_max]
        
        save_path = f'./data/{dname}_client{client_id+1}_data.pkl'
        with open(save_path, 'wb') as pkl:
            joblib.dump(client_data, pkl)
        print(f"Saved {save_path}")

if __name__ == '__main__':
    process_federated_data('./data/NYC.npy', './data/NYC_POI.npy', 'NYC', num_clients=2)
