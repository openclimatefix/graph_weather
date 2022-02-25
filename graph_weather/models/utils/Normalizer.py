from torch import tensor, zeros, ones, sum, max, sqrt, float
from torch.nn import Module, Parameter

#to do: convert to a pytorch-compatible layer, modify MeshGraphNets.py (see BatchNorm for an example)
# and use incremental update formulas for mean/std and/or exponential decay averaging
class Normalizer(Module):  
    def __init__(self, 
            size, 
            max_accumulations=10**7, 
            epsilon=1e-8,
            device=None):

        '''
        Online normalization module

        size: feature dimension
        max_accumulation: maximum number of batches
        epsilon: std cutoff for constant variable
        device: pytorch device
        '''

        super(Normalizer, self).__init__()

        self.max_accumulations = max_accumulations
        self.epsilon = epsilon

        # self.register_buffer('acc_count', tensor(0, dtype=float, device=device))
        # self.register_buffer('num_accumulations', tensor(0, dtype=float, device=device))
        # self.register_buffer('acc_sum', zeros(size, dtype=float, device=device))
        # self.register_buffer('acc_sum_squared', zeros(size, dtype=float, device=device))

        self.register_buffer('acc_count', tensor(1.0, dtype=float, device=device))
        self.register_buffer('num_accumulations', tensor(1.0, dtype=float, device=device))
        self.register_buffer('acc_sum', ones(size, dtype=float, device=device))
        self.register_buffer('acc_sum_squared', ones(size, dtype=float, device=device))

    def forward(self, batched_data, accumulate=True):
        """
        Updates mean/standard deviation and normalizes input data

        batched_data: batch of data
        accumulate: if True, update accumulation statistics
        """
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)
        
        return (batched_data - self._mean().to(batched_data.device)) / self._std().to(batched_data.device)
    
    def inverse(self, normalized_batch_data):
        """
        Unnormalizes input data
        """

        return normalized_batch_data * self._std().to(normalized_batch_data.device) + self._mean().to(normalized_batch_data.device)
          
    def _accumulate(self, batched_data):
        """
        Accumulates statistics for mean/standard deviation computation
        """
        count = tensor(batched_data.shape[0]).float()
        data_sum = sum(batched_data, dim=0)
        squared_data_sum = sum(batched_data**2, dim=0)
        
        self.acc_sum += data_sum.to(self.acc_sum.device)
        self.acc_sum_squared += squared_data_sum.to(self.acc_sum_squared.device)
        self.acc_count += count.to(self.acc_count.device)
        self.num_accumulations += 1
        
    def _mean(self):
        '''
        Returns accumulated mean
        '''
        safe_count = max(self.acc_count, tensor(1.).float())

        return self.acc_sum / safe_count
    
    def _std(self):
        '''
        Returns accumulated standard deviation
        '''
        safe_count = max(self.acc_count, tensor(1.).float())
        std = sqrt(self.acc_sum_squared / safe_count - self._mean()**2)
        
        std[std < self.epsilon] = 1.0

        return std