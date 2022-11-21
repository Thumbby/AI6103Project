import torch
import torch.nn as nn

# Positional encoding

class Embedder:
    def __init__(self,
                 max_L_log2, 
                 L, 
                 include_input=True, 
                 input_dimensions=3, 
                 periodic_fcs=[torch.sin, torch.cos]):
        '''The position encoding object

        Arguments:
        max_L_log2: the max frequency
        L: the number of the frequency
        include_input: whether return the input
        input_dimensions: the dimensions of the input data
        periodic_fcs: the preodic functions
        '''
        self.max_L_log2 = max_L_log2
        self.L = L
        self.include_input = include_input
        self.input_dimensions = input_dimensions
        self.periodic_fcs = periodic_fcs

        self.create_embedding_fc()
        
    def create_embedding_fc(self):
        embed_fcs = []
        d = self.input_dimensions
        output_dimension = 0
        if self.include_input:
            embed_fcs.append(lambda x : x)
            output_dimension += d
            
        max_L_log2 = self.max_L_log2
        N_freqs = self.L
        
        # log_sampling is true
        freq_bands = 2.**torch.linspace(0., max_L_log2, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fc in self.periodic_fcs:
                embed_fcs.append(lambda x, p_fc=p_fc, freq=freq : p_fc(x * freq))
                output_dimension += d
                    
        self.embed_fcs = embed_fcs
        self.output_dimension = output_dimension
        
    def embed(self, inputs):
        return torch.cat([fc(inputs) for fc in self.embed_fcs], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return nn.Identity(), 3
        
    embedder_obj = Embedder(max_L_log2=multires-1, L=multires)
    return embedder_obj.embed, embedder_obj.output_dimension