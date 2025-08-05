import torch
from torch.optim import Optimizer

class Viterbi(Optimizer):
    def __init__(self, params, lr=1e-3, beam_size=3, betas=(0.9, 0.8, 0.7), lambda_trans=0.1):
        # if not 0.0 <= lr:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # if beam_size < 1:
        #     raise ValueError(f"Invalid beam size: {beam_size}")
        # if not all(0.0 <= b <= 1.0 for b in betas):
        #     raise ValueError(f"Betas must be between 0 and 1")
        # if lambda_trans < 0.0:
        #     raise ValueError(f"Invalid lambda_trans: {lambda_trans}")

        defaults = dict(lr=lr, beam_size=beam_size, betas=betas, lambda_trans=lambda_trans)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['beam'] = [{'direction': torch.zeros_like(p), 'score': 0.0}]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beam_size = group['beam_size']
            betas = group['betas']
            lambda_trans = group['lambda_trans']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ViterbiOptimizer does not support sparse gradients')

                state = self.state[p]
                beam = state['beam']

                # Generate all possible candidates
                candidates = []
                for prev_candidate in beam:
                    prev_dir = prev_candidate['direction']
                    prev_score = prev_candidate['score']

                    for beta in betas:
                        # Compute new direction
                        new_dir = beta * prev_dir + (1 - beta) * grad

                        # Compute transition score (cosine similarity)
                        flat_prev = prev_dir.flatten()
                        flat_new = new_dir.flatten()
                        norm_prev = torch.norm(flat_prev)
                        norm_new = torch.norm(flat_new)
                        if norm_prev == 0 or norm_new == 0:
                            cos_sim = torch.tensor(0.0, device=grad.device)
                        else:
                            cos_sim = torch.dot(flat_prev, flat_new) / (norm_prev * norm_new)
                        transition_score = lambda_trans * cos_sim.item()

                        # Compute emission score (-grad : new_dir)
                        emission = -torch.dot(grad.flatten(), new_dir.flatten()).item()

                        total_score = prev_score + transition_score + emission

                        candidates.append({
                            'direction': new_dir.clone(),
                            'score': total_score
                        })

                # Select top beam_size candidates
                candidates.sort(key=lambda x: x['score'], reverse=True)
                new_beam = candidates[:beam_size]

                # Update beam state
                state['beam'] = new_beam

                # Update parameters using the best candidate's direction
                if new_beam:
                    best_dir = new_beam[0]['direction']
                    p.data.add_(best_dir, alpha=-lr)

        return loss