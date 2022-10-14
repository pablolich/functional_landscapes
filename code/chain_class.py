class Chain:
    
    def __init__(self, T, x, ssq, rate_acc):
        self.T = T #chain temperature
        self.x = x #vector of parameters being minimized
        self.ssq = ssq #sum squared of error
        self.rate_acc = rate_acc #acceptance rate

    def perturb(self, new_ssq):
        #calculate probability of acceptance of perturbation
        p_acc = acceptance_probability(self.ssq_vec[-1], new_ssq, self.T)
        #decide whether or not to accept
        should_swap = np.random.binomial(1, p_acc)
        if should_swap: 
