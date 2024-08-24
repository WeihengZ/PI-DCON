import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

# Define physics-informed loss loss
def darcy_loss(u, x_coor, y_coor):
    '''
    PDE residual = u_xx + u_yy + 10, where 10 is the constant uniform forcing term
    '''

    # define loss
    mse = nn.MSELoss()

    # compute pde residual
    u_x = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    pde_residual = u_xx + u_yy + 10
    pde_loss = mse(pde_residual, torch.zeros_like(pde_residual))

    return pde_loss

# define a function for visualization of the predicted function over the domain
def plot(xcoor, ycoor, f):

    # Create a scatter plot with color mapped to the 'f' values
    # you can change the plotting setting here
    plt.scatter(xcoor, ycoor, c=f, cmap='viridis', marker='o', s=10)

    # Add a colorbar
    plt.colorbar(label='f')

# define the function for model testing
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

    # split the coordinate into (x,y)
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    # model testing
    mean_relative_L2 = 0
    num = 0
    max_relative_err = -1
    min_relative_err = np.inf
    for (par, out) in loader:

        # move the batch data to device
        batch = par.shape[0]
        par = par.float().to(device)
        out = out.float().to(device)

        # model forward
        pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        L2_relative = (torch.norm(pred-out, dim=-1) / torch.norm(out, dim=-1))

        # find the max and min error sample in this batch
        max_err, max_err_idx = torch.topk(L2_relative, 1)
        if max_err > max_relative_err:
            max_relative_err = max_err
            worst_f = pred[max_err_idx,:].detach().cpu().numpy()
            worst_gt = out[max_err_idx,:].detach().cpu().numpy()
        min_err, min_err_idx = torch.topk(-L2_relative, 1)
        min_err = -min_err
        if min_err < min_relative_err:
            min_relative_err = min_err
            best_f = pred[min_err_idx,:].detach().cpu().numpy()
            best_gt = out[min_err_idx,:].detach().cpu().numpy()

        # compute average error
        mean_relative_L2 += torch.sum(L2_relative)
        num += par.shape[0]

    mean_relative_L2 /= num
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    # make the coordinates to numpy
    coor_x = test_coor_x[0].detach().cpu().numpy()
    coor_y = test_coor_y[0].detach().cpu().numpy()

    # compute appropriate color bar range
    max_color = np.amax([np.amax(worst_gt), np.amax(best_gt)])
    min_color = np.amin([np.amin(worst_gt), np.amin(best_gt)])

    # make a plot
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15,8), dpi=400)
    plt.subplot(2,3,1)
    plt.scatter(coor_x, coor_y, c=worst_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.colorbar()
    plt.title('Prediction (worst case)', fontsize=15)
    plt.subplot(2,3,2)
    plt.scatter(coor_x, coor_y, c=worst_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('Ground Truth (worst case)', fontsize=15)
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.scatter(coor_x, coor_y, c=np.abs(worst_f-worst_gt), cmap=cm, vmin=0, vmax=max_color, marker='o', s=5)
    plt.title('Absolute Error (worst case)', fontsize=15)
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.scatter(coor_x, coor_y, c=best_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.colorbar()
    plt.title('Prediction (best case)', fontsize=15)
    plt.subplot(2,3,5)
    plt.scatter(coor_x, coor_y, c=best_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('Ground Truth (best case)', fontsize=15)
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.scatter(coor_x, coor_y, c=np.abs(best_f-best_gt), cmap=cm, vmin=0, vmax=max_color, marker='o', s=5)
    plt.title('Absolute Error (best case)', fontsize=15)
    plt.colorbar()
    plt.savefig(r'../res/plots/sample_{}_{}.png'.format(args.model, args.data))

    return mean_relative_L2

# define the function for model validation
def val(model, loader, coors, device):
    '''
    Input:
        model: the model instance to be tested
        loader: validation loader of the dataset
        coors: A set of fixed coordinate
        device: cpu or gpu
    Ouput:
        mean_relative_L2: average relative error of the model prediction
    '''

    # split the coordinate into (x,y)
    test_coor_x = coors[:, 0].unsqueeze(0).float().to(device)
    test_coor_y = coors[:, 1].unsqueeze(0).float().to(device)

    # model testing
    mean_relative_L2 = 0
    num = 0
    for (par, out) in loader:

        # move the batch data to device
        batch = par.shape[0]
        par = par.float().to(device)
        out = out.float().to(device)

        # model forward
        pred = model(test_coor_x.repeat(batch,1), test_coor_y.repeat(batch,1), par)
        L2_relative = (torch.norm(pred-out, dim=-1) / torch.norm(out, dim=-1))
        mean_relative_L2 += torch.sum(L2_relative)
        num += par.shape[0]

    mean_relative_L2 /= num
    mean_relative_L2 = mean_relative_L2.detach().cpu().item()

    return mean_relative_L2

# define the function for model training
def train(args, config, model, device, loaders, coors, BC_flags):
    '''
    Input:
        args: usig this information to assign name for the output plots
        config: store the configuration for model training and testing
        model: model instance to be trained
        device: cpu or gpu
        loaders: a tuple to store (train_loader, val_loader, test_loader)
        coors: A set of fixed coordinate in the shape of (M,2)
        BC_flags: A set of binary number for the boundary indicator
            - BC_flags[i] == 1 means that coors[i,:] is the coordinate on the boundary

    '''

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
    BC_coors = coors[np.where(BC_flags==1)[0],:].float().to(device)
    pde_coors = coors[np.where(BC_flags==0)[0],:]
    num_pde_nodes = pde_coors.shape[0]

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    # define loss
    mse = nn.MSELoss()

    # visual frequency, define the number of epoch for each evaluation
    vf = config['train']['visual_freq']

    # store the train loss
    pde_avg_loss = np.inf
    bc_avg_loss = np.inf

    # move the model to the defined device
    model = model.to(device)

    # try loading the pre-trained model
    try:
        model.load_state_dict(torch.load(r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model), map_location=device))   
    except:
        print('No pre-trained model found.')

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        for e in pbar:
            
            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err = val(model, val_loader, coors, device)
                print('Best L2 relative error:', err)
                print('current period pde loss:', pde_avg_loss/vf)
                print('current period bc loss:', bc_avg_loss/vf)
                pde_avg_loss = 0
                bc_avg_loss = 0

                # save the model if new lowest validation err is seen
                if err < min_val_err:
                    torch.save(model.state_dict(), r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model))
                    min_val_err = err

            # train one epoch
            model.train()
            for (par, out) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # training with sampled coordinates
                    ss_index = np.random.randint(0, num_pde_nodes, config['train']['coor_sampling_size'])
                    pde_sampled_coors = pde_coors[ss_index, :]

                    # prepare data
                    batch = par.shape[0]
                    par = par.float().to(device)
                    BC_gt = out[:, np.where(BC_flags==1)[0]].float().to(device)
                    pde_sampled_coors_r = pde_sampled_coors.unsqueeze(0).repeat(batch, 1, 1).float().to(device)
                    bc_sampled_coors_r = BC_coors.unsqueeze(0).repeat(batch, 1, 1).float().to(device)

                    # forward to get the BC prediction
                    BC_pred = model(bc_sampled_coors_r[:,:,0], bc_sampled_coors_r[:,:,1], par)

                    # define the differentiable variables
                    sampled_x_coors = Variable(pde_sampled_coors_r[:,:,0].type(torch.FloatTensor), requires_grad=True).to(device)
                    sampled_y_coors = Variable(pde_sampled_coors_r[:,:,1].type(torch.FloatTensor), requires_grad=True).to(device)

                    # model forward
                    u_pred = model(sampled_x_coors, sampled_y_coors, par)

                    # compute the loss
                    pde_loss = darcy_loss(u_pred, sampled_x_coors, sampled_y_coors)
                    bc_loss = mse(BC_pred, BC_gt)

                    total_loss = pde_loss + config['train']['bc_weight'] * bc_loss

                    # store the loss
                    pde_avg_loss += pde_loss.detach().cpu().item()
                    bc_avg_loss += bc_loss.detach().cpu().item()

                    # update parameters
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

    # final model test
    model.load_state_dict(torch.load(r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)))   
    model.eval()
    err = test(model, test_loader, coors, device, args)
    print('Best L2 relative error on test loader:', err)