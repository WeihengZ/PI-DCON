import torch.nn as nn
import math
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch


# Physics-informed loss
def struct_loss(u, v, x_coor, y_coor, params):

    # extract parameters
    E, mu = params
    G = E / 2 / (1+mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    eps_xy = (u_y + v_x)

    # compute stress
    # sigma_xx = 2 * G * eps_xx + (E * mu / (1+mu)/ (1-2*mu)) * (eps_xx + eps_yy + eps_zz)
    # sigma_yy = 2 * G * eps_yy + (E * mu / (1+mu)/ (1-2*mu)) * (eps_xx + eps_yy + eps_zz)  # E * (eps_yy + mu * (eps_xx + eps_zz))
    # sigma_zz = 2 * G * eps_zz + (E * mu / (1+mu)/ (1-2*mu)) * (eps_xx + eps_yy + eps_zz)  # E * (eps_zz + mu * (eps_xx + eps_yy))
    sigma_xx = (E / (1-mu**2)) * (eps_xx + mu*(eps_yy))
    sigma_yy = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sigma_xy = G * eps_xy

    # compute residual
    rx = torch.autograd.grad(outputs=sigma_xx, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx),create_graph=True)[0] +\
         torch.autograd.grad(outputs=sigma_xy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy),create_graph=True)[0]
    ry = torch.autograd.grad(outputs=sigma_xy, inputs=x_coor, grad_outputs=torch.ones_like(sigma_xx),create_graph=True)[0] +\
         torch.autograd.grad(outputs=sigma_yy, inputs=y_coor, grad_outputs=torch.ones_like(sigma_xy),create_graph=True)[0]

    return rx, ry

# Neumann Boundation condition loss
def bc_edgeY_loss(u, v, x_coor, y_coor, params):

    # extract parameters
    E, mu = params
    G = E / 2 / (1+mu)

    # compute strain
    eps_xx = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    eps_yy = torch.autograd.grad(outputs=v, inputs=y_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    v_x = torch.autograd.grad(outputs=v, inputs=x_coor, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    eps_xy = (u_y + v_x)
    
    # compute stress
    sigma_yy = (E / (1-mu**2)) * (eps_yy + mu*(eps_xx))
    sigma_xy = G * eps_xy

    return sigma_yy, sigma_xy

# function for ploting the predicted function
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=15)
    # Add a colorbar
    plt.colorbar(label='f')

# function for testing
def test(model, loader, coors, device, args):
    '''
    Input:
        model: the model instance to be tested
        loader: testing loader of the dataset
        coors: A set of fixed coordinate
        device: cpu or gpu
        args: usig this information to assign name for the output plots
    Ouput:
        A plot of the PDE solution ground-truth, prediction, and absolute error
    '''

    # split the coordinates
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    mean_relative_L2 = 0
    num = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (par, u, v) in loader:

        # move the data to device
        batch = par.shape[0]
        par = par.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)

        # model forward
        u_pred, v_pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        L2_relative = torch.sqrt(torch.sum((u_pred-u)**2 + (v_pred-v)**2, -1)) / torch.sqrt(torch.sum((u)**2 + (v)**2, -1))

        # find the max and min error sample in this batch
        max_err, max_err_idx = torch.topk(L2_relative, 1)
        if max_err > max_relative_err:
            max_relative_err = max_err
            worst_f = u_pred[max_err_idx,:].detach().cpu().numpy()
            worst_gt = u[max_err_idx,:].detach().cpu().numpy()
        min_err, min_err_idx = torch.topk(-L2_relative, 1)
        min_err = -min_err
        if min_err < min_relative_err:
            min_relative_err = min_err
            best_f = u_pred[min_err_idx,:].detach().cpu().numpy()
            best_gt = u[min_err_idx,:].detach().cpu().numpy()

        # store mean relative error
        mean_relative_L2 += torch.sum(L2_relative).detach().cpu().item()
        num += u.shape[0]

    mean_relative_L2 /= num

    # make the coordinates to numpy
    coor_x = test_coor_x[0].detach().cpu().numpy()
    coor_y = test_coor_y[0].detach().cpu().numpy()

    # color bar range
    max_color = np.amax([np.amax(worst_gt), np.amax(best_gt)])
    min_color = np.amin([np.amin(worst_gt), np.amin(best_gt)])

    # make a plot
    SS = 20
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15,8), dpi=400)
    plt.subplot(2,3,1)
    plt.scatter(coor_x, coor_y, c=worst_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=SS)
    plt.colorbar()
    plt.title('Prediction (worst case)', fontsize=15)
    plt.subplot(2,3,2)
    plt.scatter(coor_x, coor_y, c=worst_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=SS)
    plt.title('Ground Truth (worst case)', fontsize=15)
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.scatter(coor_x, coor_y, c=np.abs(worst_f-worst_gt), cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=SS)
    plt.title('Absolute Error (worst case)', fontsize=15)
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.scatter(coor_x, coor_y, c=best_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=SS)
    plt.colorbar()
    plt.title('Prediction (best case)', fontsize=15)
    plt.subplot(2,3,5)
    plt.scatter(coor_x, coor_y, c=best_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=SS)
    plt.title('Ground Truth (best case)', fontsize=15)
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.scatter(coor_x, coor_y, c=np.abs(best_f-best_gt), cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=SS)
    plt.title('Absolute Error (best case)', fontsize=15)
    plt.colorbar()
    plt.savefig(r'../res/plots/sample_{}_{}.png'.format(args.model, args.data))

    return mean_relative_L2

def val(model, loader, coors, device):

    # split the coordinates
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    mean_relative_L2 = 0
    num = 0
    for (par, u, v) in loader:
        
        # move the data to device
        batch = par.shape[0]
        par = par.float().to(device)
        u = u.float().to(device)
        v = v.float().to(device)

        # model forward
        u_pred, v_pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        L2_relative = torch.sqrt(torch.sum((u_pred-u)**2 + (v_pred-v)**2, -1)) / torch.sqrt(torch.sum((u)**2 + (v)**2, -1))

        # compute relative error
        mean_relative_L2 += torch.sum(L2_relative)
        num += u.shape[0]

        # compute absolute error for point sampling probability computation
        abs_err = torch.mean(torch.abs(u_pred-u) + torch.abs(v_pred-v), 0).detach().cpu().numpy()
        
        # # check GPU usage
        # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

    mean_relative_L2 /= num
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    return mean_relative_L2, abs_err


# define the training function
def train(args, config, model, device, loaders, coors, flag_BCxy, flag_BCy, flag_BC_load, params):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])
    print('BC weight', config['train']['bc_weight'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # extract coors
    xy_BC_coors = coors[np.where(flag_BCxy==1)[0],:]
    y_BC_coors = coors[np.where(flag_BCy==1)[0],:]
    load_BC_coors = coors[np.where(flag_BC_load==1)[0],:]
    pde_coors = coors[np.where(flag_BC_load+flag_BCxy+flag_BCy==0)[0],:]
    num_pde_nodes = pde_coors.shape[0]

    # Move the data to device
    xy_BC_coors = xy_BC_coors.float().to(device)
    y_BC_coors = y_BC_coors.float().to(device)
    load_BC_coors = load_BC_coors.float().to(device)
    coors = coors.float().to(device)
    print('Number of PDE points:', num_pde_nodes)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    # visual frequency for evaluation and store the updated model parameters
    vf = config['train']['visual_freq']

    # move the model to the defined device
    model = model.to(device)

    # initialize recored loss values
    avg_loss1 = np.inf
    avg_loss2 = np.inf
    avg_loss3 = np.inf
    avg_loss4 = np.inf
    avg_loss5 = np.inf
    avg_loss6 = np.inf

    # try loading the pre-trained model
    try:
        model.load_state_dict(torch.load(r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model), map_location=device))   
    except:
        print('No pre-trained model found.')

    # define the training weight
    weight_bc = config['train']['bc_weight']

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        for e in pbar:
          
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err, pointwise_err = val(model, val_loader, coors, device)

                print('Best L2 relative error:', err)
                print('x-direction prescribed displacement loss', avg_loss1)
                print('y-direction prescribed displacement loss:', avg_loss2)
                print('hole prescribed displacement loss', avg_loss3)
                print('free boundary condtion loss:', avg_loss4)
                print('x-direction PDE residual loss:', avg_loss5)
                print('y-direction PDE residual loss:', avg_loss6) 
                if err < min_val_err:
                    torch.save(model.state_dict(), r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model))
                    min_val_err = err
                
                # update recored loss values
                avg_loss1 = 0
                avg_loss2 = 0
                avg_loss3 = 0
                avg_loss4 = 0
                avg_loss5 = 0
                avg_loss6 = 0

            # train one epoch
            model.train()
            for (par, u, v) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # compute the sampling probability of each collocation point
                    batchsize = u.shape[0]
                    p_pde_sampling = pointwise_err[np.where(flag_BC_load+flag_BCxy+flag_BCy==0)[0]]
                    p_pde_sampling = p_pde_sampling / np.sum(p_pde_sampling)

                    ss_index = np.random.choice(np.arange(num_pde_nodes), config['train']['coor_sampling_size'], p=p_pde_sampling)
                    pde_sampled_coors = pde_coors[ss_index, :]
                    pde_sampled_coors = pde_sampled_coors.float().to(device)

                    # prepare the data
                    par = par.float().to(device)
                    pde_sampled_coors_r = pde_sampled_coors.unsqueeze(0).repeat(batchsize,1,1)
                    xy_BC_coors_r = xy_BC_coors.unsqueeze(0).repeat(batchsize,1,1)
                    y_BC_coors_r = y_BC_coors.unsqueeze(0).repeat(batchsize,1,1)
                    load_BC_coors_r = load_BC_coors.unsqueeze(0).repeat(batchsize,1,1)

                    # forward to get the prediction on fixed boundary
                    u_BCxy_pred, v_BCxy_pred = model(xy_BC_coors_r[:,:,0], xy_BC_coors_r[:,:,1], par)

                    # forward to get the prediction on free boundary
                    x_pde_bcy = Variable(y_BC_coors_r[:,:,0], requires_grad=True)
                    y_pde_bcy = Variable(y_BC_coors_r[:,:,1], requires_grad=True)
                    u_BCy_pred, v_BCy_pred = model(x_pde_bcy, y_pde_bcy, par)
                    sigma_yy, sigma_xy = bc_edgeY_loss(u_BCy_pred, v_BCy_pred, x_pde_bcy, y_pde_bcy, params)

                    # forward to get the prediction on loading element
                    u_load_pred, v_load_pred = model(load_BC_coors_r[:,:,0], load_BC_coors_r[:,:,1], par)
                    u_load_gt = u[:,np.where(flag_BC_load==1)[0]].float().to(device)
                    v_load_gt = v[:,np.where(flag_BC_load==1)[0]].float().to(device)

                    # forward to get the prediction on pde inside element
                    x_pde = Variable(pde_sampled_coors_r[:,:,0], requires_grad=True)
                    y_pde = Variable(pde_sampled_coors_r[:,:,1], requires_grad=True)
                    u_pde_pred, v_pde_pred = model(x_pde, y_pde, par)
                    rx_pde, ry_pde = struct_loss(u_pde_pred, v_pde_pred, x_pde, y_pde, params)
                    
                    # compute the loss
                    bc_loss1 = mse(u_load_pred, u_load_gt) 
                    bc_loss2 = mse(v_load_pred, v_load_gt) 
                    bc_loss3 = torch.mean(u_BCxy_pred**2) + torch.mean(v_BCxy_pred**2) 
                    bc_loss4 = torch.mean(sigma_yy**2) + torch.mean(sigma_xy**2) 
                    pde_loss1 = torch.mean(rx_pde**2)
                    pde_loss2 = torch.mean(ry_pde**2) 
                    total_loss = (bc_loss1 + bc_loss2 + bc_loss3 + bc_loss4) + (pde_loss1 + pde_loss2) * weight_bc

                    # store the loss
                    avg_loss1 += bc_loss1.detach().cpu().item()
                    avg_loss2 += bc_loss2.detach().cpu().item()
                    avg_loss3 += bc_loss3.detach().cpu().item()
                    avg_loss4 += bc_loss4.detach().cpu().item()
                    avg_loss5 += pde_loss1.detach().cpu().item()
                    avg_loss6 += pde_loss2.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

    # final test
    model.load_state_dict(torch.load(r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)))   
    model.eval()
    err = test(model, test_loader, coors, device, args)
    print('Best L2 relative error on test loader:', err)