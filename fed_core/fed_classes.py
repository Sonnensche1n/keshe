import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_crypto.encoder import CramerEncoder
from fed_crypto.elgamal_homo import ElGamalHomo

class FedClient:
    """
    Federated Learning Client with Encryption capabilities.
    For this PoC, we simulate the interface.
    """
    def __init__(self, client_id, local_data, encoder, crypto_module, sub_private_key):
        self.client_id = client_id
        self.encoder = encoder
        self.crypto = crypto_module
        self.sub_private_key = sub_private_key
        
    def train_and_encrypt(self, local_grad):
        """
        Simulates extracting gradient, quantizing, encoding and encrypting.
        1. Quantize and Encode (Cramer).
        2. Encrypt with ElGamal.
        3. Return ciphertexts.
        """
        encoded_grad = self.encoder.encode(local_grad)
        # Encrypt (Simulated element-wise or flat encryption)
        # For a tensor, we would iterate and encrypt each element.
        # Here we do it for a single value for PoC simulation.
        c1, c2 = self.crypto.encrypt(encoded_grad.item())
        return (c1, c2)
        
    def provide_decryption_share(self, C_agg1):
        """
        Compute partial decryption for threshold decryption.
        """
        return self.crypto.partial_decrypt(C_agg1, self.sub_private_key)

class FedServer:
    """
    Federated Learning Server that aggregates encrypted gradients.
    """
    def __init__(self):
        pass
        
    def aggregate_encrypted_updates(self, encrypted_updates_list, p):
        """
        Perform homomorphic addition on ciphertexts.
        """
        return ElGamalHomo.aggregate_ciphertexts(encrypted_updates_list, p)
        
    def request_decryption_and_update(self, clients_dict, C_agg, p, q, g, crypto_module, encoder):
        """
        1. Broadcast C_agg1 to clients.
        2. Collect D_i shares.
        3. Combine shares to decrypt aggregated gradient.
        4. Decode (Cramer inverse).
        """
        partial_decryptions = {}
        for client_id, client in clients_dict.items():
            partial_decryptions[client_id] = client.provide_decryption_share(C_agg[0])
            
        decrypted_encoded_grad = crypto_module.combine_shares_and_decrypt(C_agg, partial_decryptions, p, q, g)
        
        import torch
        # Decode
        tensor_grad = torch.tensor([decrypted_encoded_grad])
        decoded_float = encoder.decode(tensor_grad)
        
        return decoded_float.item()

if __name__ == "__main__":
    import torch
    import random
    from fed_crypto.vss_keygen import generate_safe_prime, VSSKeyGen
    
    print("=== Federated Learning Crypto Pipeline Simulation ===")
    # 1. System Init
    p, q = generate_safe_prime(128)
    h_rand = random.randint(2, p - 2)
    g = pow(h_rand, (p - 1) // q, p)
    
    n_clients = 3
    threshold = 2
    
    dkg = VSSKeyGen(p, q, g, 2, n_clients, threshold)
    client_data = {i: dkg.generate_shares() for i in range(1, n_clients + 1)}
    
    S_keys = {}
    for k in range(1, n_clients + 1):
        shares = [client_data[i][1][k] for i in range(1, n_clients + 1)]
        S_keys[k] = dkg.synthesize_sub_private_key(shares)
        
    Y = dkg.generate_global_public_key([client_data[i][2] for i in range(1, n_clients + 1)])
    
    # 2. FL Setup
    encoder = CramerEncoder(scale_factor=1000, field_size=10**10)
    crypto_mod = ElGamalHomo(p, q, g, Y)
    
    clients = {}
    for i in range(1, n_clients + 1):
        clients[i] = FedClient(i, None, encoder, crypto_mod, S_keys[i])
        
    server = FedServer()
    
    # 3. Simulate Training Gradients
    grad1 = torch.tensor([-0.15])
    grad2 = torch.tensor([0.25])
    grad3 = torch.tensor([0.10])
    
    print(f"Client 1 local gradient: {grad1.item()}")
    print(f"Client 2 local gradient: {grad2.item()}")
    print(f"Client 3 local gradient: {grad3.item()}")
    
    # 4. Encrypt
    c1 = clients[1].train_and_encrypt(grad1)
    c2 = clients[2].train_and_encrypt(grad2)
    c3 = clients[3].train_and_encrypt(grad3)
    
    # 5. Server Aggregates
    C_agg = server.aggregate_encrypted_updates([c1, c2, c3], p)
    print("\nServer aggregated encrypted updates.")
    
    # 6. Threshold Decrypt & Decode
    # Only using Client 1 and 2 to meet threshold 2
    active_clients = {1: clients[1], 2: clients[2]}
    final_grad = server.request_decryption_and_update(active_clients, C_agg, p, q, g, crypto_mod, encoder)
    
    print(f"\nFinal Aggregated Decoded Gradient: {final_grad}")
    print(f"Expected: {-0.15 + 0.25 + 0.10}")
