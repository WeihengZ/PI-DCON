import torch
import torch.nn as nn

'''
Neural operator models for 2D darcy flow problem
'''
class DeepONet_darcy(nn.Module):

    def __init__(self, config, input_dim):
        super().__init__()

        # branch network
        branch_layers = [nn.Linear(input_dim, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            branch_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            branch_layers.append(nn.Tanh())
        branch_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*branch_layers)

        # trunk network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.trunk = nn.Sequential(*trunk_layers)
        
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # only extract the function values information to represent the PDE parameters
        enc = self.branch(par[...,-1])   

        # compute the PDE solution prediction
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        x = self.trunk(xy)    # (B,M,F)
        u = torch.einsum('bij,bj->bi', x, enc)    

        return u

class Improved_DeepOnet_darcy(nn.Module):

    def __init__(self, config, input_dim):
        super().__init__()

        # branch network
        self.FC1b = nn.Linear(input_dim, config['model']['fc_dim'])
        self.FC2b = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3b = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])

        # trunk network
        self.FC1c = nn.Linear(2, config['model']['fc_dim'])
        self.FC2c = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3c = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])

        # branch encoder
        self.be = nn.Sequential(nn.Linear(input_dim, config['model']['fc_dim']), nn.Tanh())
        self.ce = nn.Sequential(nn.Linear(2, config['model']['fc_dim']), nn.Tanh())

        # activation function
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # get the coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # get the embeddings
        par_emb = self.be(par[...,-1]).unsqueeze(1)   
        coor_emb = self.ce(xy)   

        # paramter forward
        enc = self.FC1b(par[...,-1]).unsqueeze(1)   
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC2b(enc)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC3b(enc)

        # coordinate forward
        xy = self.FC1c(xy)   
        xy = self.act(xy)
        xy = (1-xy) * par_emb + xy * coor_emb
        xy = self.FC2c(xy)
        xy = self.act(xy)
        xy = (1-xy) * par_emb + xy * coor_emb
        xy = self.FC3c(xy)

        # combine
        u = torch.sum(xy*enc, -1)    

        return u
    
class DCON_darcy(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        branch_layers = [nn.Linear(3, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            branch_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            branch_layers.append(nn.Tanh())
        branch_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*branch_layers)

        # trunk network
        self.FC1u = nn.Linear(2, config['model']['fc_dim'])
        self.FC2u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC4u = nn.Linear(config['model']['fc_dim'], 1)
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # get the kernel using both the coordinate and function values information
        enc = self.branch(par)
        enc = torch.amax(enc, 1, keepdim=True)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # predict u
        u = self.FC1u(xy)   
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)  
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   
        u = torch.mean(u * enc, -1)   

        return u

class New_model_darcy(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,1)
        )
    
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        u = self.fc(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)

        return u



'''
Neural operator models for 2D plate problem
'''
class fc(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Linear(dim, dim)
        self.act = nn.Tanh()
    
    def forward(self, x):
        return self.act(self.ff(x))

class DeepONet_plate(nn.Module):

    def __init__(self, config, num_input_dim):
        super().__init__()

        # branch network #1
        trunk_layers = [nn.Linear(num_input_dim, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(fc(config['model']['fc_dim']))
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch1 = nn.Sequential(*trunk_layers)

        # branch network #2
        trunk_layers = [nn.Linear(num_input_dim, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(fc(config['model']['fc_dim']))
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch2 = nn.Sequential(*trunk_layers)

        # trunk network #1
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(fc(config['model']['fc_dim']))
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.trunk1 = nn.Sequential(*trunk_layers)

        # trunk network #2
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(fc(config['model']['fc_dim']))
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.trunk2 = nn.Sequential(*trunk_layers)
        
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''
        
        # PDE parameter encoding
        enc1 = self.branch1(par[:,:,-1])    
        enc2 = self.branch2(par[:,:,-1])    

        # PDE solution prediction
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        ux = self.trunk1(xy)   
        uy = self.trunk2(xy)   
        u = torch.einsum('bij,bj->bi', ux, enc1)   
        v = torch.einsum('bij,bj->bi', uy, enc2)     

        return u, v

class Improved_DeepONet_plate(nn.Module):

    def __init__(self, config, num_input_dim):
        super().__init__()

        # branch network
        self.FC1b = nn.Linear(num_input_dim, config['model']['fc_dim'])
        self.FC2b = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3b = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])

        # trunk network
        self.FC1c1 = nn.Linear(2, config['model']['fc_dim'])
        self.FC2c1 = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3c1 = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])

        self.FC1c2 = nn.Linear(2, config['model']['fc_dim'])
        self.FC2c2 = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3c2 = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])

        # branch encoder
        self.be = nn.Sequential(nn.Linear(num_input_dim, config['model']['fc_dim']), nn.Tanh())
        self.ce = nn.Sequential(nn.Linear(2, config['model']['fc_dim']), nn.Tanh())
        self.act = nn.Tanh()

        
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # get the coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # get the embeddings
        par_emb = self.be(par[...,-1]).unsqueeze(1)   # (B,1,F)
        coor_emb = self.ce(xy)   # (B, M, F)

        # paramter forward
        enc = self.FC1b(par[...,-1]).unsqueeze(1)   # (B,F)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC2b(enc)
        enc = self.act(enc)
        enc = (1-enc) * par_emb + enc * coor_emb
        enc = self.FC3b(enc)

        # u coordinate forward
        xy1 = self.FC1c1(xy)   # (B,M,F)
        xy1 = self.act(xy1)
        xy1 = (1-xy1) * par_emb + xy1 * coor_emb
        xy1 = self.FC2c1(xy1)
        xy1 = self.act(xy1)
        xy1 = (1-xy1) * par_emb + xy1 * coor_emb
        xy1 = self.FC3c1(xy1)
        # combine
        u = torch.sum(xy1*enc, -1)    # (B,M)

        # u coordinate forward
        xy2 = self.FC1c2(xy)   # (B,M,F)
        xy2 = self.act(xy2)
        xy2 = (1-xy2) * par_emb + xy2 * coor_emb
        xy2 = self.FC2c2(xy2)
        xy2 = self.act(xy2)
        xy2 = (1-xy2) * par_emb + xy2 * coor_emb
        xy2 = self.FC3c2(xy2)
        # combine
        v = torch.sum(xy2*enc, -1)    # (B,M)
        
        return u, v

class DCON_plate(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(3, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # branch network
        self.lift = nn.Linear(3, config['model']['fc_dim_branch'])
        self.val_lift = nn.Linear(1, config['model']['fc_dim_branch'])

        # trunk network 1
        self.FC1u = nn.Linear(2, config['model']['fc_dim'])
        self.FC2u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3u = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])

        # trunk network 2
        self.FC1v = nn.Linear(2, config['model']['fc_dim'])
        self.FC2v = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.FC3v = nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'])
        self.act = nn.Tanh()
        
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        # get the kernel
        enc = self.branch(par)  
        enc = torch.amax(enc, 1, keepdim=True)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # predict u
        u = self.FC1u(xy)   
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)   
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)   
        u = torch.mean(u * enc, -1)    # (B, M)

        # predict v
        v = self.FC1v(xy)   
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)  
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)   
        v = torch.mean(v * enc, -1)    # (B, M)

        return u, v

class New_model_plate(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,1)
        )
    
    def forward(self, x_coor, y_coor, par):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, N, 3): boundary coordinates and the function values, each row is (x,y,u) of one collocation point
                           N is the total number of BC points
        
        model output:
            u (B, M): PDE solution fucntion values over the whole domain
        '''

        u = self.fc1(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)
        v = self.fc2(torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)).squeeze(-1)

        return u, v