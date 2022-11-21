import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

# Gaussian Activation Function
def gaussian(x, mean=0, std=0.1):
    gauss = torch.exp((-(x - mean) ** 2)/(2* std ** 2))
    return gauss

# Model Architecture of NeRF
class NeRF(nn.Module):
    def __init__(self, 
                 depth=4, 
                 width=256, 
                 pos_in=3, 
                 views_in=3, 
                 skip_conn=[4], 
                 act_fc = 'relu'):
        '''Initialize NeRF model
        Augments:
            act_fc: type of activation function used in MLP
            depth: number of MLP layers 
            width: width of MLP layers
            pos_in: channel of sample position, 3 (x,y,z) by default, much higher with positional encoding
            views_in: channel of sample views, similar 
            skip_conn: We add a skip connection to the fifth MLP layer by default, 
                    but if the depth <= 4, then no skip connection will be applied.
        '''
        super(NeRF, self).__init__()
        self.depth = depth
        self.width = width
        self.pos_ch = pos_in
        self.views_ch = views_in
        self.skip_conn = skip_conn
        self.act_fc = act_fc

        self.mlp_layers = nn.ModuleList(
            [nn.Linear(pos_in, width)] + [nn.Linear(width, width) if i not in self.skip_conn else nn.Linear(width + pos_in, width) for i in range(self.depth-1)])
        
        self.view_layers = nn.ModuleList([nn.Linear(views_in + width, int(width/2))])

        self.feature_layer = nn.Linear(width, width)
        self.sigma_layer = nn.Linear(width, 1)
        self.rgb_layer = nn.Linear(int(width/2), 3)
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.pos_ch, self.views_ch], dim=-1)
        h = input_pts
        for i, _ in enumerate(self.mlp_layers):
            h = self.mlp_layers[i](h)

            #apply selected activation function
            if self.act_fc == 'relu':
                h = F.relu(h)
            elif self.act_fc == 'tanh':
                h = F.tanh(h)
            elif self.act_fc == 'siren':
                h = torch.sin(h)
            elif self.act_fc == 'gaussian':
                h = gaussian(h)
            if i in self.skip_conn:
                h = torch.cat([input_pts, h], -1)

        sigma = self.sigma_layer(h)
        feature = self.feature_layer(h)
        h = torch.cat([feature, input_views], -1)
        
        for i, l in enumerate(self.view_layers):
            h = self.view_layers[i](h)
            if self.act_fc == 'relu':
                h = F.relu(h)
            elif self.act_fc == 'tanh':
                h = F.tanh(h)
            elif self.act_fc == 'siren':
                h = torch.sin(h)
            elif self.act_fc == 'gaussian':
                h = gaussian(h)

        color= self.rgb_layer(h)
        outputs = torch.cat([color, sigma], -1)

        return outputs
