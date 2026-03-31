import torch
import copy

class STANServer:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def aggregate(self, client_weights_list, client_data_sizes):
        """
        FedAvg Aggregation
        """
        total_data = sum(client_data_sizes)
        global_state = self.global_model.state_dict()
        
        # Initialize an empty state dict with zeros for accumulation
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            
        # Accumulate weighted client models
        for client_weights, data_size in zip(client_weights_list, client_data_sizes):
            weight_ratio = data_size / total_data
            for key in global_state.keys():
                global_state[key] += client_weights[key] * weight_ratio
                
        # Update global model
        self.global_model.load_state_dict(global_state)
        return self.global_model.state_dict()
        
    def get_global_model_state(self):
        return self.global_model.state_dict()
