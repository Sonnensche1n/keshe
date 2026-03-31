import random
from Crypto.Util.number import inverse

class ElGamalHomo:
    """
    Additive Homomorphic ElGamal Encryption.
    Supports encryption of Cramer-encoded gradients and Server-side aggregation.
    """
    def __init__(self, p, q, g, Y):
        """
        :param p: large prime
        :param q: order of the subgroup
        :param g: generator
        :param Y: Global Public Key
        """
        self.p = p
        self.q = q
        self.g = g
        self.Y = Y
        
    def encrypt(self, m):
        """
        Encrypts an integer message m using exponential ElGamal for additive homomorphism.
        r = random(1, q-1)
        C1 = g^r mod p
        C2 = g^m * Y^r mod p
        Returns (C1, C2)
        """
        r = random.randint(1, self.q - 1)
        C1 = pow(self.g, r, self.p)
        
        # C2 = g^m * Y^r mod p
        gm = pow(self.g, m, self.p)
        Yr = pow(self.Y, r, self.p)
        C2 = (gm * Yr) % self.p
        
        return (C1, C2)
        
    @staticmethod
    def aggregate_ciphertexts(ciphers_list, p):
        """
        Server aggregates multiple ciphertexts.
        C_agg1 = prod(C_i1) mod p
        C_agg2 = prod(C_i2) mod p
        Returns (C_agg1, C_agg2)
        """
        C_agg1 = 1
        C_agg2 = 1
        for C1, C2 in ciphers_list:
            C_agg1 = (C_agg1 * C1) % p
            C_agg2 = (C_agg2 * C2) % p
        return (C_agg1, C_agg2)
        
    def partial_decrypt(self, C_agg1, S_i):
        """
        Client generates partial decryption.
        D_i = (C_agg1)^S_i mod p
        """
        return pow(C_agg1, S_i, self.p)
        
    def combine_shares_and_decrypt(self, C_agg, partial_decryptions_dict, p, q, g):
        """
        Server combines partial decryptions D_i using Lagrange interpolation to find Y^r.
        Then finds g^m = C_agg2 / Y^r mod p.
        Finally, uses baby-step giant-step or exhaustive search to find m from g^m.
        """
        C_agg1, C_agg2 = C_agg
        
        # 1. Lagrange interpolation to find Y^r = prod(D_i ^ lambda_i) mod p
        Yr = 1
        S = list(partial_decryptions_dict.keys()) # list of client indices k
        
        for i, D_i in partial_decryptions_dict.items():
            # Calculate lambda_i = prod_{j!=i} (0-j)/(i-j) mod q
            num = 1
            den = 1
            for j in S:
                if i != j:
                    num = (num * (0 - j)) % q
                    den = (den * (i - j)) % q
            
            lambda_i = (num * inverse(den, q)) % q
            
            term = pow(D_i, lambda_i, p)
            Yr = (Yr * term) % p
            
        # 2. Recover g^m = C_agg2 * (Y^r)^-1 mod p
        Yr_inv = inverse(Yr, p)
        gm = (C_agg2 * Yr_inv) % p
        
        # 3. Discrete Log to find m (Since m is a sum of gradients, it shouldn't be astronomically large)
        # Using simple exhaustive search for PoC. For production, use Baby-step Giant-step.
        # We assume m is positive after Cramer encoding, but it could be very large if it's L + grad
        # Actually, because of Cramer encoding, the sum m might be > L.
        # This is a limitation of Exponential ElGamal: m MUST be small enough to brute-force.
        # For this PoC, we will implement a simple brute-force bounded search.
        
        # Since m could be negative originally, it was encoded to L + m.
        # So m could be near L (which is huge, e.g., 10**10).
        # We need a Baby-Step Giant-Step or Pollard's rho for huge M,
        # OR we can just use Paillier for real-world additive homomorphic properties where DLOG isn't needed.
        # But since the requirement explicitly asks for exponential ElGamal:
        # For this PoC to work in reasonable time, we will bound our search around 0 and around L.
        
        # We check m_pos and current_gm_pos in parallel
        # Let's fix the loop logic
        
        m_pos = 0
        m_neg = 0
        current_gm_pos = 1
        current_gm_neg = 1
        g_inv = inverse(g, p)
        max_search = 1000000 
        
        if gm == 1:
            return 0
            
        # Optimization: We check if the sum wrapped around q (the group order)
        # Because we used Cramer transform, the exponent is sum(M_i). 
        # If some were negative, they were L + m_neg.
        # Sum = k*L + sum(m). This goes into the exponent: g^{k*L + sum(m)} mod p
        # To make DLOG work, we should just search for sum(m) directly since L is known.
        # But actually, we don't know k. 
        # Let's just search g^{sum(m)} and if it matches, return that sum(m).
        # We can reconstruct the Cramer encoding later or just return the negative sum.
        
        while m_pos < max_search:
            m_pos += 1
            m_neg -= 1
            
            current_gm_pos = (current_gm_pos * g) % p
            if current_gm_pos == gm:
                return m_pos
                
            current_gm_neg = (current_gm_neg * g_inv) % p
            if current_gm_neg == gm:
                # We return the negative exponent directly because we disabled Cramer L addition
                return m_neg
                
        raise ValueError(f"Discrete log failed! Value is outside search bounds. (m_pos={m_pos}, gm={gm})")

if __name__ == "__main__":
    from vss_keygen import generate_safe_prime, VSSKeyGen
    
    print("Testing ElGamal Homomorphic Encryption...")
    p, q = generate_safe_prime(128)
    h_rand = random.randint(2, p - 2)
    g = pow(h_rand, (p - 1) // q, p)
    
    n_clients = 3
    threshold = 2
    
    # 1. Setup Keys
    dkg = VSSKeyGen(p, q, g, 2, n_clients, threshold)
    client_data = {i: dkg.generate_shares() for i in range(1, n_clients + 1)}
    
    S_keys = {}
    for k in range(1, n_clients + 1):
        shares = [client_data[i][1][k] for i in range(1, n_clients + 1)]
        S_keys[k] = dkg.synthesize_sub_private_key(shares)
        
    Y = dkg.generate_global_public_key([client_data[i][2] for i in range(1, n_clients + 1)])
    
    # 2. Encrypt
    elgamal = ElGamalHomo(p, q, g, Y)
    m1 = 15
    m2 = 25
    c1 = elgamal.encrypt(m1)
    c2 = elgamal.encrypt(m2)
    
    # 3. Aggregate
    c_agg = ElGamalHomo.aggregate_ciphertexts([c1, c2], p)
    
    # 4. Partial Decrypt (threshold = 2)
    D1 = elgamal.partial_decrypt(c_agg[0], S_keys[1])
    D2 = elgamal.partial_decrypt(c_agg[0], S_keys[2])
    
    partial_decryptions = {1: D1, 2: D2}
    
    # 5. Combine and Decrypt
    decrypted_m = elgamal.combine_shares_and_decrypt(c_agg, partial_decryptions, p, q, g)
    
    print(f"Original m1: {m1}, m2: {m2}")
    print(f"Decrypted Aggregation (m1 + m2): {decrypted_m}")
    print(f"Success: {decrypted_m == m1 + m2}")
