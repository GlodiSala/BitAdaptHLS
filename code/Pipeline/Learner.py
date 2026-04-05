import torch
from Pipeline.Energy import analyze_and_compute_energy

class Learner:
    def __init__(self, args, fp32_energy_ref=1.0, fp32_sr_ref=0.0, method=None):
        if method:
            self.method = method
        self.args = args
        self.device = torch.device(
            self.args.device if torch.cuda.is_available() else "cpu"
        )
        self.fp32_energy_ref = fp32_energy_ref
        self.fp32_sr_ref     = fp32_sr_ref

    def rate_calculator_3d(self, precoder, csi):
        # calculate sum rate
        # Handle 4D tensors - squeeze singleton dimensions if needed
        if csi.dim() == 4:
            csi = csi.squeeze()
        if precoder.dim() == 4:
            precoder = precoder.squeeze()
        
        # Calculate W using corrected einsum equation
        W = torch.einsum("nij,njk->nik", torch.conj(csi), precoder)
        diag_W = torch.diagonal(torch.abs(W) ** 2, dim1=1, dim2=2)
        SINR = diag_W / (torch.sum(torch.abs(W) ** 2, 2) - diag_W + self.args.noise_pwr)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
    
        return sumRate

    def ceil_pass(self, n,gamma=0.01):
        n_ceil = n.ceil()
        n_grad = n + (gamma * n)
        return (n_ceil - n_grad).detach() + n_grad
    

    def criterium_with_bitpruning(self, FDP, channel, quantizers=None, model=None, lambda_reg=None):
        """
        Compute loss for adaptive bit-width optimization.
        Optimizes Energy Efficiency (EE = sum_rate / energy) to automatically balance:
        - Low sum_rate → increase bits for better performance
        - Low EE → decrease bits for better efficiency
        """
        # Task-specific loss (sum-rate)
        FDP = FDP / torch.linalg.norm(FDP, dim=(1, 2), keepdim=True)
        sum_rate_loss = self.rate_calculator_3d(FDP, channel).mean()

        if model is not None and quantizers is not None and lambda_reg is not None:
            # Energy-based regularization with separate weight and activation quantization
            Q_weights = []
            Q_activations = []
            for q in quantizers.values():
                # Weight bit: average in_proj and out_proj bits for MHA, or just weight bit otherwise
                if hasattr(q['weight'], 'bit'):
                    w_bit = self.ceil_pass(q['weight'].bit)
                    if 'weight_out' in q and hasattr(q['weight_out'], 'bit'):
                        w_bit_out = self.ceil_pass(q['weight_out'].bit)
                        w_bit = (w_bit + w_bit_out) / 2  # average for energy estimate
                else:
                    w_bit = 16
                Q_weights.append(w_bit)

                if hasattr(q['activation'], 'bit'):
                    act_bit_value = self.ceil_pass(q['activation'].bit)
                else:
                    act_bit_value = 16
                Q_activations.append(act_bit_value)
            
            total_energy = analyze_and_compute_energy(model, Q=Q_weights, Q_activations=Q_activations, input_size=(1, 64, 128)) * 1e-6
            #total_energy = analyze_and_compute_energy(model, Q=Q_weights, input_size=(1, 4, 128)) * 1e-6
            # ADAPTIVE BIT ALLOCATION: Optimize Energy Efficiency (EE = sum_rate / energy)
            # This automatically balances performance vs efficiency:
            # - When EE is low (high energy, low sum_rate), gradients push bits DOWN
            # - When sum_rate is low, gradients push bits UP to improve performance
            # - The model learns optimal bit allocation per layer based on sensitivity
            
            # Energy Efficiency loss (minimize negative EE = maximize EE)
            # EE = sum_rate / energy, so loss = -EE = -sum_rate / energy
            #energy_efficiency = sum_rate_loss / (total_energy + 1e-8)  # Add epsilon for stability
            energy_efficiency =  total_energy/ self.fp32_energy_ref
            # Combined loss: maximize sum_rate AND maximize EE
            # lambda_reg controls the trade-off between raw performance and efficiency

            ee_loss = -lambda_reg * energy_efficiency
            
            # Calcul de l'Efficacité Énergétique (Numériquement stable)
            
            # On veut MAXIMISER le Sum Rate (donc on le met en négatif)
            fp32_sr = self.fp32_sr_ref if self.fp32_sr_ref > 0 else sum_rate_loss.detach()

            
            # Total loss: maximize sum_rate weighted by energy efficiency
            # This encourages high sum_rate but penalizes excessive energy consumption
            return (sum_rate_loss / fp32_sr) + ee_loss
        else:
            return sum_rate_loss

    def criterium_FDP(self, FDP, channel):
        FDP = FDP / torch.linalg.norm(FDP, dim=(1, 2), keepdim=True)
        sum_rate = self.rate_calculator_3d(FDP, channel)
        return sum_rate.mean()
