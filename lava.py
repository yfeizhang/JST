from otdd.pytorch.datasets import load_torchvision_data_shuffle
from otdd.pytorch.distance_fast import DatasetDistance, FeatureCost
import matplotlib.pyplot as plt
import numpy as np
import time

# Corrupted will return list of indices that were corrupted
def load_data_corrupted(corrupt_type='feature', dataname=None, data=None, valid_size=0, random_seed=2024, resize=None,
                                        stratified=True, shuffle=False, 
                                        training_size=None, test_size=None, currupt_por=0, std_dev=None):
    if corrupt_type == 'feature':
        loaders, full_dict, shuffle_ind  = load_torchvision_data_shuffle(dataname, valid_size=valid_size, 
                                                                        random_seed=random_seed, 
                                                                        resize = resize, stratified=stratified, 
                                                                        shuffle=shuffle, maxsize=training_size, 
                                                                        maxsize_test = test_size, corrupted_per=currupt_por, std_dev=std_dev, data=data)
    else:
        shuffle_ind = []
        loaders, full_dict  = load_torchvision_data_shuffle(dataname, valid_size=valid_size, random_seed=random_seed, 
                                                            resize = resize, stratified=stratified, shuffle=shuffle,
                                                            maxsize=training_size, maxsize_test = test_size, shuffle_per=0, data=data) 
    
    return loaders, shuffle_ind, full_dict


# Get list of all indices of a dataset (subset)
def get_indices(singleloader):
    return singleloader.batch_sampler.sampler.indices

def train_with_corrupt_flag(trainloader, shuffle_ind, train_indices):
    trained_with_flag = []
    itr = 0
    counting_labels = {} # For statistics
    for trai in trainloader:
        #print(trai)
        train_images = trai[0]
        train_labels = trai[1]
        # get one image of the training from that batch
        for i in range(len(train_labels)):
            train_image = train_images[i]
            train_label = train_labels[i]
            trained_with_flag.append([train_image,train_label, train_indices[itr] in shuffle_ind])
            itr = itr + 1
            if train_label.item() in counting_labels:
                counting_labels[train_label.item()] += 1
            else:
                counting_labels[train_label.item()] = 1
    return trained_with_flag

# Get dual solution of OT problem
def get_OT_dual_sol(trainloader, testloader, dist_method=None, training_size=10000, resize=32, dim = 3):
    feature_cost = FeatureCost(src_embedding = None,
                               src_dim = (dim,1,resize),
                               tgt_embedding = None,
                               tgt_dim = (dim,1,resize),
                               p = 2,
                               device='cuda', method=dist_method)

    dist = DatasetDistance(trainloader, testloader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-1,
                           device='cuda')

    tic = time.perf_counter()
    dual_sol = dist.dual_sol(maxsamples = training_size, return_coupling = True)
    # π = dist.compute_coupling(entreg = entreg, m=m, method=method, nb_dummies=nb_dummies)
    toc = time.perf_counter()
    print(f"distance calculation takes {toc - tic:0.4f} seconds")

    for i in range(len(dual_sol)):
        dual_sol[i] = dual_sol[i].to('cpu')
    return dual_sol

def compute_dual(trainloader, testloader, dist_method, training_size, shuffle_ind, dim, resize=32):
    # to return 2
    # get indices of corrupted and non corrupted for visualization
    train_indices = get_indices(trainloader)
    trained_with_flag = train_with_corrupt_flag(trainloader, shuffle_ind, train_indices)
    
    # to return 1
    # OT Dual calculation
    dual_sol = get_OT_dual_sol(trainloader, testloader, dist_method=dist_method, resize=resize, training_size=training_size, dim=dim)
    return dual_sol, trained_with_flag, train_indices


# For VISUALIZATION

# Get the calibrated gradient of the dual solution
# Which can be considered as data values (more in paper...)
def values(dual_sol, training_size):
    f1k = np.array(dual_sol[0].squeeze())

    trainGradient = [0]*training_size
    trainGradient = (1+1/(training_size-1))*f1k - sum(f1k)/(training_size-1)
    return list(trainGradient)

# Sort the calibrated values and keep original indices 
# Higher value is worse
def sort_and_keep_indices(trainGradient):
    my_array = np.array(trainGradient)
    sorted_gradient_ind = np.argsort(my_array)
    sorted_gradient_ind = [np.array([index]) for index in sorted_gradient_ind]
    return sorted_gradient_ind
    
# Visualize based on sorted values (calibrated gradient)
# Prints 3 graphs, with a random baselines (explained in paper...)
def visualize_values_distr_sorted(tdid, tsidx, trsize, portion, trainGradient, reverse=False):
    x1, y1, base = [], [], []
    
    if reverse == True:
        tsidx.reverse()
        
    for vari in range(10,trsize,10):
        found_none = sum(tdid[tsidx[i][0]][2] for i in range(vari))
        
#             print('inspected: '+str(vari), 'found: '+str(found),  
#                   'detection rate: ', str(found / poisoned), 'baseline = '+str(vari*0.2*0.9))
        
        print(f'inspected: {vari}, found: {vari-found_none} detection rate: {(vari-found_none) / (vari):.2f} baseline: {1-portion}')
            
        x1.append(vari)
        y1.append((vari-found_none) / (vari))
        base.append(1-portion)
    plt.scatter(x1, y1, s=10)
    plt.scatter(x1, base, s=10)
    # naming the x axis
    plt.xlabel('Inspected Images')
    # naming the y axis
    plt.ylabel('Detected Images')
    plt.yticks([0,1])

    # giving a title to my graph
    plt.title('Detection vs Gradient Inspection')

    # function to show the plot
    plt.show()
    
    if reverse == True:
        tsidx.reverse()

    ################# GETTING POISON FLAG WITH GRADIENT ############
    x, y = [],[]
    poison_cnt = 0
    last_ind = -1
    x_poisoned = []
    non_poisoned = []
    for i in range(trsize):
        oriid = tsidx[i][0]
        x.append(trainGradient[oriid])
        #print(trainGradient[i])
        y.append(tdid[oriid][2])
        poison_cnt += 1 if tdid[oriid][2] else 0
        last_ind = oriid if tdid[oriid][2] else last_ind
        if tdid[oriid][2]:
            x_poisoned.append(trainGradient[oriid])
        else:
            non_poisoned.append(trainGradient[oriid])
    plt.scatter(x, y, s=10)

    # naming the x axis
    plt.xlabel('Gradient')
    # naming the y axis
    plt.ylabel('Poisoned Image')
    plt.yticks([0,1])

    # giving a title to my graph
    plt.title('Gradient vs Poisoned')

    # function to show the plot
    plt.show()

    print("number of poisoned images", poison_cnt)
    print("last index of poison", last_ind)

    ########################### HISTOGRAM PLOT #################################################
    tminElement = np.amin(trainGradient)
    tmaxElement = np.amax(trainGradient)
    bins = np.linspace(tminElement, tmaxElement,200)
    n_non_poisoned, bins_np, _ = plt.hist(non_poisoned, bins,label="Clean Images")
    n_x_poisoned, bins_xp, _ = plt.hist(x_poisoned, bins,label="Poisoned Images", edgecolor='None', alpha = 0.5,)
    # naming the x axis
    plt.xlabel('Gradient')
    # naming the y axis
    plt.ylabel('Number of Images')
    plt.title('Gradient of Poisoned and Non-Poisoned Images Lambda=(1,1)')
    plt.legend(loc="upper left")
    plt.show()
    
    plt.plot(bins_np[:-1], np.cumsum(n_non_poisoned), label="Clean Images")
    plt.plot(bins_xp[:-1], np.cumsum(n_x_poisoned), label="Poisoned Images")
    plt.plot(bins_xp[:-1], np.cumsum(n_non_poisoned) + np.cumsum(n_x_poisoned), label="Total Images")
    # naming the x axis
    plt.xlabel('Gradient')
    # naming the y axis
    plt.ylabel('Number of Images')
    plt.title('Cumulative Number of Poisoned and Non-Poisoned Images')
    plt.legend(loc="upper left")
    plt.show()
    
    return n_non_poisoned, bins_np, n_x_poisoned, bins_xp, y1, base

# Get the data values and also visualizes the detection of 'bad' data
def compute_values_and_visualize(dual_sol, trained_with_flag, training_size, portion, reverse=False):
    calibrated_gradient = values(dual_sol, training_size)
    sorted_gradient_ind = sort_and_keep_indices(calibrated_gradient)
    n_non_poisoned, bins_np, n_x_poisoned, bins_xp, y1, base = visualize_values_distr_sorted(trained_with_flag, sorted_gradient_ind, training_size, portion, calibrated_gradient, reverse)
    return calibrated_gradient, sorted_gradient_ind, n_non_poisoned, bins_np, n_x_poisoned, bins_xp, y1, base