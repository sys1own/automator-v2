import hashlib
import concurrent.futures
from typing import List, Tuple, Optional, Callable

class PietrzakVDFEngine:
    """
    Implements Pietrzak's Verifiable Delay Function.
    Features dynamic sub-sequence checkpoint generation and logarithmic-time verification.
    """
    def __init__(self):
        # Secure 2048-bit RSA Modulus N of unknown order.
        self.N = int("""11578135832726543888390737039572635939225883210404099710334814885542721869399430162597401211181734978711467439568979313264426508933499426297370008064998781938183182606558661642283995642998632612669359306050361048869818816766467389470921865434407873204968846101967209935105268028308738308182747183063529452093539820718696894080775532598370701831498186107878368792070624956326792476575196500802058098315570265219468165561110034237142071536417743519548684713781745290680072671676356779435889797379659357422955519808388904323675005852504620067347967399990810775836894229272895645517173926868616110996144829377484606771891""")

    def _derive_projection_challenge(self, x: int, y: int, mu: int) -> int:
        payload = f"{x}:{y}:{mu}".encode('utf-8')
        h = hashlib.sha256(payload).hexdigest()
        return int(h, 16) % self.N

    def evaluate_and_prove(self, seed: bytes, T: int) -> Tuple[int, List[int]]:
        assert (T & (T - 1)) == 0, "Steps parameter T must be a power of 2."
        x = int.from_bytes(hashlib.sha256(seed).digest(), 'big') % self.N
        
        # 1. Compute sequential terminal value y
        curr = x
        for _ in range(T):
            curr = pow(curr, 2, self.N)
        y = curr
        
        proof = []
        
        # 2. Recursive halving proof generation with dynamic midpoint generation
        def build_proof_recursive(x_in: int, y_in: int, t_steps: int):
            if t_steps == 1:
                return
            half_t = t_steps // 2
            
            # Correctly compute the midpoint of the CURRENT active sequence
            mid_val = x_in
            for _ in range(half_t):
                mid_val = pow(mid_val, 2, self.N)
                
            proof.append(mid_val)
            r = self._derive_projection_challenge(x_in, y_in, mid_val)
            
            # Form coordinates for next recursive verification layer
            x_next = (pow(x_in, r, self.N) * mid_val) % self.N
            y_next = (pow(mid_val, r, self.N) * y_in) % self.N
            
            build_proof_recursive(x_next, y_next, half_t)

        build_proof_recursive(x, y, T)
        return y, proof

    def verify_proof(self, seed: bytes, T: int, y: int, proof: List[int]) -> bool:
        x = int.from_bytes(hashlib.sha256(seed).digest(), 'big') % self.N
        
        def verify_recursive(x_val: int, y_val: int, t_steps: int, p_idx: int) -> bool:
            if t_steps == 1:
                return pow(x_val, 2, self.N) == y_val
            if p_idx >= len(proof):
                return False
                
            mu = proof[p_idx]
            r = self._derive_projection_challenge(x_val, y_val, mu)
            
            x_next = (pow(x_val, r, self.N) * mu) % self.N
            y_next = (pow(mu, r, self.N) * y_val) % self.N
            
            return verify_recursive(x_next, y_next, t_steps // 2, p_idx + 1)

        return verify_recursive(x, y, T, 0)

class AsynchronousPacingQueue:
    """Manages non-blocking verification of VDF proofs using a background thread pool."""
    def __init__(self, max_threads: int = 4):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
        self.engine = PietrzakVDFEngine()
        self.active_jobs: List[concurrent.futures.Future] = []

    def submit_verification(self, round_id: int, seed: bytes, T: int, y: int, proof: List[int], callback: Callable[[int, bool], None]):
        def task_wrapper() -> Tuple[int, bool]:
            success = self.engine.verify_proof(seed, T, y, proof)
            return round_id, success

        future = self.pool.submit(task_wrapper)
        
        def process_completed_future(fut: concurrent.futures.Future):
            try:
                r_id, status = fut.result()
                callback(r_id, status)
            except Exception:
                callback(round_id, False)

        future.add_done_callback(process_completed_future)
        self.active_jobs.append(future)

    def prune_active_jobs(self):
        self.active_jobs = [f for f in self.active_jobs if not f.done()]

    def shutdown(self):
        self.pool.shutdown(wait=True)
