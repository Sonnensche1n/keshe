import random
from Crypto.Util.number import getPrime, isPrime

def generate_safe_prime(bits):
    """Generate a safe prime p = 2q + 1 where q is also prime."""
    while True:
        q = getPrime(bits - 1)
        p = 2 * q + 1
        if isPrime(p):
            return p, q

class Polynomial:
    """Represents a polynomial for Secret Sharing over Z_q."""
    def __init__(self, degree, secret, q):
        self.q = q
        # f(x) = a_0 + a_1*x + ... + a_{t-1}*x^{t-1} mod q
        # a_0 is the secret
        self.coeffs = [secret] + [random.randint(1, q - 1) for _ in range(degree)]
        
    def evaluate(self, x):
        """Evaluate polynomial at point x modulo q."""
        result = 0
        for i, coeff in enumerate(self.coeffs):
            result = (result + coeff * pow(x, i, self.q)) % self.q
        return result

class VSSKeyGen:
    """
    Distributed Key Generation using Pedersen/Feldman VSS.
    Works over group G of order q, which is a subgroup of Z_p*.
    """
    def __init__(self, p, q, g, h, num_clients, threshold):
        self.p = p
        self.q = q
        self.g = g
        self.h = h  # Optional, mainly for Pedersen. We'll use Feldman style mostly for public keys.
        self.n = num_clients
        self.t = threshold
        
    def generate_shares(self):
        """
        Client generates their polynomial and distributes shares/commitments.
        Returns: 
            secret_a0: The client's main secret share a_{i,0}
            shares: list of (k, s_{i,k}) where s_{i,k} = f_i(k) mod q
            commitments: list of C_{i,j} = g^{a_{i,j}} mod p
        """
        secret_a0 = random.randint(1, self.q - 1)
        poly = Polynomial(self.t - 1, secret_a0, self.q)
        
        # 1. Generate shares for each client k = 1...n
        shares = {}
        for k in range(1, self.n + 1):
            shares[k] = poly.evaluate(k)
            
        # 2. Generate commitments C_{i,j} = g^{a_{i,j}} mod p (Feldman VSS)
        # If we use Pedersen: C_{i,j} = g^{a_{i,j}} h^{b_{i,j}} mod p (requires another polynomial for b)
        # We'll use Feldman here as it directly provides the public key Y components.
        commitments = [pow(self.g, coeff, self.p) for coeff in poly.coeffs]
        
        return secret_a0, shares, commitments
        
    def verify_share(self, k, share_value, commitments):
        """
        Client k verifies the received share from client i against client i's commitments.
        share_value: s_{i,k}
        commitments: [C_{i,0}, ..., C_{i, t-1}]
        Check: g^{s_{i,k}} == prod_{j=0}^{t-1} (C_{i,j})^{k^j} mod p
        """
        lhs = pow(self.g, share_value, self.p)
        
        rhs = 1
        for j, C_ij in enumerate(commitments):
            exp = pow(k, j, self.q)
            term = pow(C_ij, exp, self.p)
            rhs = (rhs * term) % self.p
            
        return lhs == rhs
        
    def synthesize_sub_private_key(self, valid_shares):
        """
        Client k sums all received valid shares to get S_k.
        S_k = sum(s_{i,k}) mod q
        """
        S_k = sum(valid_shares) % self.q
        return S_k
        
    def generate_global_public_key(self, all_client_commitments):
        """
        Synthesize global public key Y from constant term commitments C_{i,0}.
        Y = prod(C_{i,0}) mod p = prod(g^{a_{i,0}}) = g^{sum(a_{i,0})}
        """
        Y = 1
        for commitments in all_client_commitments:
            C_i0 = commitments[0]
            Y = (Y * C_i0) % self.p
        return Y

if __name__ == "__main__":
    print("Testing Distributed Key Generation (DKG)...")
    # Generate small parameters for testing
    p, q = generate_safe_prime(128)
    
    # Find generator g of order q in Z_p*
    while True:
        h_rand = random.randint(2, p - 2)
        g = pow(h_rand, (p - 1) // q, p)
        if g != 1:
            break
            
    h = 2 # Dummy h, not strictly used in Feldman
    
    n_clients = 3
    threshold = 2
    
    dkg = VSSKeyGen(p, q, g, h, n_clients, threshold)
    
    # 1. All clients generate polynomials and shares
    client_data = {}
    all_commitments = []
    
    for i in range(1, n_clients + 1):
        secret, shares, commitments = dkg.generate_shares()
        client_data[i] = {'shares_to_send': shares, 'commitments': commitments, 'secret': secret}
        all_commitments.append(commitments)
        
    # 2. Distribute shares and verify
    client_sub_keys = {}
    for k in range(1, n_clients + 1):
        received_valid_shares = []
        for i in range(1, n_clients + 1):
            share_from_i = client_data[i]['shares_to_send'][k]
            commitments_from_i = client_data[i]['commitments']
            
            is_valid = dkg.verify_share(k, share_from_i, commitments_from_i)
            assert is_valid, f"Share from {i} to {k} is invalid!"
            received_valid_shares.append(share_from_i)
            
        # 3. Calculate sub private key S_k
        client_sub_keys[k] = dkg.synthesize_sub_private_key(received_valid_shares)
        
    # 4. Generate global public key Y
    Y = dkg.generate_global_public_key(all_commitments)
    
    # Verify mathematically: Y should be g^{sum(a_{i,0})} mod p
    total_secret = sum([client_data[i]['secret'] for i in range(1, n_clients + 1)]) % q
    Y_expected = pow(g, total_secret, p)
    
    print(f"Global Public Key Y generated successfully: {Y == Y_expected}")
    print(f"Client 1 sub-key S_1: {client_sub_keys[1]}")
