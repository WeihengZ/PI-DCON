import scipy.io as sio
import numpy as np
import yaml
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from models_plane import DON, Att_coor2, DON2, DON2b, IDON
import argparse

# define arguements
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--data', type=str, default='plane_dis_high')
parser.add_argument('--model', type=str, default='IDON')
args = parser.parse_args()
print('Model forward phase: {}'.format(args.phase))
print('Using dataset: {}'.format(args.data))
print('Using model: {}'.format(args.model))

# load the data
mat_contents = sio.loadmat(r'./data/{}.mat'.format(args.data))
u = mat_contents['final_u'] # (B, M)
v = mat_contents['final_v'] # (B, M)
coor = mat_contents['coors'] # (M, 2)
flag_BCxy = mat_contents['flag_BCxy']  # (M, 1)
flag_BCy = mat_contents['flag_BCy']  # (M, 1)
flag_load = mat_contents['flag_BC_load']  # (M, 1)

# load the factor
scalar_factor = 1e-4
youngs = mat_contents['young'][0][0] * scalar_factor
nu = mat_contents['poisson'][0][0]

# structure the parameter input
id_param = np.where(flag_load==1)[0]
datasize = u.shape[0]
params = np.concatenate((np.repeat(np.expand_dims(coor[id_param,:],0),datasize,axis=0),
                         np.expand_dims(u[:,id_param],-1)), -1)    # (B, M', 3)
num_bc_nodes = params.shape[1]

# extract configuration
with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
if args.model == 'Trans':
    # normalize the data
    # coor = (coor - np.min(coor, 0, keepdims=True)) / (np.max(coor, 0, keepdims=True) - np.min(coor, 0, keepdims=True))
    model = DON2b(config, [80, 50, 1], num_bc_nodes, device)
if args.model == 'DON':
    model = DON(config, num_bc_nodes)
if args.model == 'IDON':
    model = IDON(config, num_bc_nodes)
if args.model == 'PINN':
    model = DON(config, num_bc_nodes)

# define dataset
params = torch.tensor(params)    # (B, N, 3)
u = torch.tensor(u)    # (B, M)
v = torch.tensor(v)    # (B, N)
coors = torch.tensor(coor)    # (M,2)

# define data loader
if args.model == 'PINN':
    bar1 = [0,1]
    bar2 = [0,1]
    bar3 = [0,1]
else:
    bar1 = [0,int(0.7*datasize)]
    bar2 = [int(0.7*datasize),int(0.8*datasize)]
    bar3 = [int(0.8*datasize),int(datasize)]
train_dataset = torch.utils.data.TensorDataset(params[bar1[0]:bar1[1],:,:], u[bar1[0]:bar1[1],:], v[bar1[0]:bar1[1],:])
val_dataset = torch.utils.data.TensorDataset(params[bar2[0]:bar2[1],:,:], u[bar2[0]:bar2[1],:], v[bar2[0]:bar2[1],:])
test_dataset = torch.utils.data.TensorDataset(params[bar3[0]:bar3[1],:,:], u[bar3[0]:bar3[1],:], v[bar3[0]:bar3[1],:])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batchsize'], shuffle=False)

# PINO loss
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

def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=15)
    # Add a colorbar
    plt.colorbar(label='f')

def test(model, loader, coors):

    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    mean_relative_L2 = 0
    num = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (par, u, v) in loader:
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
        mean_relative_L2 += torch.sum(L2_relative)
        num += u.shape[0]

    mean_relative_L2 /= num
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

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
    plt.savefig(r'./res/plots/sample_{}_{}.png'.format(args.model, args.data))

    return mean_relative_L2

def val(model, loader, coors):

    test_coor_x = coors[:, 0].unsqueeze(0)
    test_coor_y = coors[:, 1].unsqueeze(0)

    mean_relative_L2 = 0
    num = 0
    for (par, u, v) in loader:

        batch = par.shape[0]
        par = par.float().to(device)
        u = u.float()
        v = v.float()
        # model forward
        u_pred, v_pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        u_pred = u_pred.detach().cpu()
        v_pred = v_pred.detach().cpu()
        L2_relative = torch.sqrt(torch.sum((u_pred-u)**2 + (v_pred-v)**2, -1)) / torch.sqrt(torch.sum((u)**2 + (v)**2, -1))

        # compute pointwise error
        abs_err = torch.mean(torch.abs(u_pred-u) + torch.abs(v_pred-v), 0).detach().cpu().numpy()
        mean_relative_L2 += torch.sum(L2_relative)
        num += u.shape[0]
        
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

    xy_BC_coors = xy_BC_coors.float().to(device)
    y_BC_coors = y_BC_coors.float().to(device)
    load_BC_coors = load_BC_coors.float().to(device)
    coors = coors.float().to(device)
    print('Number of PDE points:', num_pde_nodes)

    # define optimizer and loss
    mse = nn.MSELoss()
    lr=config['train']['base_lr']
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    # visual frequency
    vf = config['train']['visual_freq']

    # err history
    err_hist = []

    # store the train loss
    avg_loss1 = 10
    avg_loss2 = 10
    avg_loss3 = 10
    avg_loss4 = 10
    avg_loss5 = 10
    avg_loss6 = 10
    lambda_value = config['train']['bc_weight']

    # move the model to the defined device
    model = model.to(device)

    # initialize the loss weight
    weight1 = 1
    weight2 = 1
    weight3 = 1
    weight4 = 1
    weight5 = 1
    weight6 = 1

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        for e in pbar:
          
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err, pointwise_err = val(model, val_loader, coors)
                err_hist.append(err)
                plt.figure()
                plt.plot(err_hist)
                plt.savefig(r'./res/logs/err_hist_{}_{}'.format(args.data, args.model))

                # update the weight
                avg_loss1 = avg_loss1 / vf
                avg_loss2 = avg_loss2 / vf
                avg_loss3 = avg_loss3 / vf
                avg_loss4 = avg_loss4 / vf
                avg_loss5 = avg_loss5 / vf
                avg_loss6 = avg_loss6 / vf

                digit1 = math.floor(math.log(avg_loss1, 10))
                digit2 = int(math.floor(math.log(avg_loss2, 10)))
                digit3 = int(math.floor(math.log(avg_loss3, 10)))
                digit4 = int(math.floor(math.log(avg_loss4, 10)))
                digit5 = int(math.floor(math.log(avg_loss5, 10)))
                digit6 = int(math.floor(math.log(avg_loss6, 10)))
                digit_min = np.array([digit1, digit2, digit3, digit4, digit5, digit6])
                digit_min = np.min(digit_min)

                # weight1 = 1 * 10 ** (digit_min-digit1)
                # weight2 = 1 * 10 ** (digit_min-digit2)
                # weight3 = 1 * 10 ** (digit_min-digit3)
                # weight4 = 1 * 10 ** (digit_min-digit4)
                # weight5 = 1 * 10 ** (digit_min-digit5)
                # weight6 = 1 * 10 ** (digit_min-digit6)
                print('weights:', weight1, weight2, weight3, weight4, weight5, weight6)

                print('Best L2 relative error:', err)
                print('current period loss #1:', avg_loss1)
                print('current period loss #2:', avg_loss2)
                print('current period loss #3:', avg_loss3)
                print('current period loss #4:', avg_loss4)
                print('current period loss #5:', avg_loss5)
                print('current period loss #6:', avg_loss6)

                print('current lambda values:', lambda_value)
                pde_avg_loss = 0
                bc_avg_loss = 0
                if err < min_val_err:
                    torch.save(model.state_dict(), r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model))
                    min_val_err = err

            # initialize loss weight
            if e % vf == 1000:
                w1 = [1e-10]
                w2 = [1e-10]
                w3 = [1e-10]
                w4 = [1e-10]
                w5 = [1e-10]
                w6 = [1e-10]

            # train one epoch
            model.train()
            for (par, u, v) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # training with sampled coordinates
                    batchsize = u.shape[0]
                    p_pde_sampling = pointwise_err[np.where(flag_BC_load+flag_BCxy+flag_BCy==0)[0]]
                    # p_pde_sampling = np.ones_like(p_pde_sampling)
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
                    total_loss = weight1*bc_loss1 + weight2*bc_loss2 + weight3*bc_loss3 +\
                                 weight4*bc_loss4 + weight5*pde_loss1 + weight6*pde_loss2

                    # store the gradient
                    if e % vf == 1000:

                        optimizer.zero_grad()
                        for p in model.parameters():
                            p.grad_prev = 0

                        (bc_loss1).backward(retain_graph=True)
                        norm_bc1 = 0
                        for p in model.parameters():
                            if p.grad != None:
                                norm_bc1 += torch.sum((p.grad - p.grad_prev)**2)
                                p.grad_prev = p.grad
                        norm_bc1 = torch.sqrt(norm_bc1)
                        w1.append(norm_bc1.detach().cpu().item())
                        

                        (bc_loss2).backward(retain_graph=True)
                        norm_bc2 = 0
                        for p in model.parameters():
                            if p.grad != None:
                                norm_bc2 += torch.sum((p.grad - p.grad_prev)**2)
                                p.grad_prev = p.grad
                        norm_bc2 = torch.sqrt(norm_bc2)
                        w2.append(norm_bc2.detach().cpu().item())

                        (bc_loss3).backward(retain_graph=True)
                        norm_bc3 = 0
                        for p in model.parameters():
                            if p.grad != None:
                                norm_bc3 += torch.sum((p.grad - p.grad_prev)**2)
                                p.grad_prev = p.grad
                        norm_bc3 = torch.sqrt(norm_bc3)
                        w3.append(norm_bc3.detach().cpu().item())

                        (bc_loss4).backward(retain_graph=True)
                        norm_bc4 = 0
                        for p in model.parameters():
                            if p.grad != None:
                                norm_bc4 += torch.sum((p.grad - p.grad_prev)**2)
                                p.grad_prev = p.grad
                        norm_bc4 = torch.sqrt(norm_bc4)
                        w4.append(norm_bc4.detach().cpu().item())

                        (pde_loss1).backward(retain_graph=True)
                        norm_pde1 = 0
                        for p in model.parameters():
                            if p.grad != None:
                                norm_pde1 += torch.sum((p.grad - p.grad_prev)**2)
                                p.grad_prev = p.grad
                        norm_pde1 = torch.sqrt(norm_pde1)
                        w5.append(norm_pde1.detach().cpu().item())

                        (pde_loss2).backward(retain_graph=True)
                        norm_pde2 = 0
                        for p in model.parameters():
                            if p.grad != None:
                                norm_pde2 += torch.sum((p.grad - p.grad_prev)**2)
                                p.grad_prev = p.grad
                        norm_pde2 = torch.sqrt(norm_pde2)
                        w6.append(norm_pde2.detach().cpu().item())

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


            # adjust loss weight
            if e % vf == 1000:
                r1 = sum(w1) / len(w1)
                r2 = sum(w2) / len(w2)
                r3 = sum(w3) / len(w3)
                r4 = sum(w4) / len(w4)
                r5 = sum(w5) / len(w5)
                r6 = sum(w6) / len(w6)
                mean_sum = r1 + r2 + r3 + r4 + r5 + r6
                weight1 = mean_sum / r1
                weight2 = mean_sum / r2
                weight3 = mean_sum / r3
                weight4 = mean_sum / r4
                weight5 = mean_sum / r5
                weight6 = mean_sum / r6
                print('weight:', weight1, weight2, weight3, weight4, weight5, weight6)

    # final test
    model.load_state_dict(torch.load(r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)))   
    model.eval()
    err = test(model, test_loader, coors)
    print('Best L2 relative error on test loader:', err)

train(args, config, model, device, (train_loader, val_loader, test_loader), coors, flag_BCxy, flag_BCy, flag_load, [youngs, nu])


