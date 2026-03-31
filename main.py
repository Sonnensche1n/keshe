import torch
import joblib
import time
import matplotlib.pyplot as plt
import numpy as np

from models import Model
from client import STANClient
from server import STANServer
from utils import AggregatedEvaluator
from load import device

def plot_metrics(records):
    epochs = records['epoch']
    acc_valid = np.array(records['acc_valid'])
    acc_test = np.array(records['acc_test'])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i, k in enumerate([1, 5, 10, 20]):
        plt.plot(epochs, acc_valid[:, i], label=f'valid_acc@{k}')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, k in enumerate([1, 5, 10, 20]):
        plt.plot(epochs, acc_test[:, i], label=f'test_acc@{k}')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('federated_metrics.png')
    print("Metrics plot saved to federated_metrics.png")

def main():
    dname = 'NYC'
    num_clients = 2
    num_rounds = 5
    local_epochs = 1
    
    print("=== Step 1: Loading Federated Data ===")
    clients_data = []
    global_u_max, global_l_max = 0, 0
    
    for i in range(num_clients):
        file_path = f'./data/{dname}_client{i+1}_data.pkl'
        try:
            with open(file_path, 'rb') as f:
                data = joblib.load(f)
                clients_data.append(data)
                global_u_max = max(global_u_max, data[6])
                global_l_max = max(global_l_max, data[7])
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Please run federated_data.py first.")
            return

    # Extract mat2s from the first client (it's the same for all)
    mat2s = torch.FloatTensor(clients_data[0][2]).to(device)

    # Calculate global bounding box for coordinates
    # ex = (max_lat, min_lat, max_lon, min_lon)
    max_lat = max([data[1][:, :, :, 0].max() for data in clients_data])
    min_lat = min([data[1][:, :, :, 0].min() for data in clients_data])
    max_lon = max([data[1][:, :, :, 1].max() for data in clients_data])
    min_lon = min([data[1][:, :, :, 1].min() for data in clients_data])
    global_ex = (max_lat, min_lat, max_lon, min_lon)

    print(f"Global u_max: {global_u_max}, l_max: {global_l_max}")

    print("\n=== Step 2: Initializing Models & Server/Clients ===")
    hours = 24 * 7 # from load.py
    global_model = Model(t_dim=hours+1, l_dim=global_l_max+1, u_dim=global_u_max+1, 
                         embed_dim=50, ex=global_ex, dropout=0)
    
    server = STANServer(global_model)
    
    clients = []
    for i in range(num_clients):
        data = clients_data[i]
        # data = [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max]
        # We need (trajs, mat1, mat2s, mat2t, labels, lens)
        user_data = (data[0], data[1], mat2s, data[3], data[4], data[5])
        client = STANClient(client_id=i+1, user_data=user_data, model=Model(t_dim=hours+1, l_dim=global_l_max+1, u_dim=global_u_max+1, embed_dim=50, ex=global_ex, dropout=0))
        clients.append(client)
        
    print("\n=== Step 3: Starting Federated Training ===")
    records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
    
    # We will use all clients for evaluation
    eval_user_data_list = [(data[0], data[1], mat2s, data[3], data[4], data[5]) for data in clients_data]
    evaluator = AggregatedEvaluator(server.global_model, eval_user_data_list, mat2s)

    for round_idx in range(num_rounds):
        print(f"\n--- Communication Round {round_idx + 1}/{num_rounds} ---")
        start_time = time.time()
        
        global_state = server.get_global_model_state()
        client_weights_list = []
        client_data_sizes = []
        
        # 1. Local Training
        for client in clients:
            print(f"Client {client.client_id} training locally...")
            weights, data_size = client.local_train(global_state, epochs=local_epochs)
            client_weights_list.append(weights)
            client_data_sizes.append(data_size)
            
        # 2. Server Aggregation
        print("Server aggregating updates...")
        server.aggregate(client_weights_list, client_data_sizes)
        
        # 3. Evaluation
        print("Evaluating global model...")
        acc_valid, acc_test = evaluator.evaluate()
        
        records['epoch'].append(round_idx + 1)
        records['acc_valid'].append(acc_valid)
        records['acc_test'].append(acc_test)
        
        print(f"Round {round_idx + 1} completed in {time.time() - start_time:.2f}s")
        
    plot_metrics(records)
    print("\nFederated Learning Training Finished!")

if __name__ == '__main__':
    main()
