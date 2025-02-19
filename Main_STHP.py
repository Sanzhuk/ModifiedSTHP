import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils
import sys

from preprocess.Dataset import get_dataloader
from transformer.Models_STHP import SparseTransformer
from tqdm import tqdm
import math


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')
       
    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    
    return trainloader, testloader, num_types

def get_count_model_tensor(time_data, event_data, bin_size):
    # Get max time across all sequences in the batch
    mx = torch.max(time_data)
    
    L = torch.ceil(mx / bin_size).to(torch.long).to(time_data.device)
    # Use the actual number of event types from the model
    K = 12  # This should match num_types in your model
    batch_size = time_data.size(0)
    
    # Change to include batch dimension
    C = torch.zeros(batch_size, L, K, device=time_data.device)
    
    # Process each sequence in the batch
    for b in range(batch_size):
        time_seq = time_data[b]
        event_seq = event_data[b]
        
        for i in range(len(time_seq)):
            if time_seq[i] == 0:  # Skip padding
                continue
            interval_idx = (torch.ceil(time_seq[i] / bin_size) - 1).to(torch.long)
            event_idx = event_seq[i] - 1  # Convert to 0-based indexing
            if event_idx < K:  # Add bounds check
                C[b, interval_idx, event_idx] += 1
    
    return C

def get_volume_model_tensor(time_data, volume_data, bin_size_volume):
    mx = torch.max(time_data)
    L = torch.ceil(mx / bin_size_volume).to(torch.long).to(time_data.device)
    batch_size = time_data.size(0)
    
    # Change dimensions from (batch_size, L, 1) to (batch_size, 1, L)
    C = torch.zeros(batch_size, L, 1, device=time_data.device)
    C_count = torch.zeros(batch_size, L, 1, device=time_data.device)
    
    # Process each sequence in the batch
    for b in range(batch_size):
        time_seq = time_data[b]
        volume_seq = volume_data[b]
        
        for i in range(len(time_seq)):
            if time_seq[i] == 0:  # Skip padding
                continue
            interval_idx = (torch.ceil(time_seq[i] / bin_size_volume) - 1).to(torch.long)
            # Update indexing to match tensor dimensions
            C[b, interval_idx, 0] += volume_seq[i]
            C_count[b, interval_idx, 0] += 1
    
    # Avoid division by zero
    C_count[C_count == 0] = 1
    # C = C / C_count
    
    return C

def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        
        """ prepare data """
        event_time, time_gap, event_type, event_volume = map(lambda x: x.to(opt.device), batch)
        
        count_matrix = get_count_model_tensor(event_time, event_type, model.bin_size)
        volume_matrix = get_volume_model_tensor(event_time, event_volume, model.bin_size_volume)
        # event_time shape: (1, 4000)
        
        """ forward """
        optimizer.zero_grad()
        # Resets gradient so that the old gradient is not taken into account
        
        enc_out, prediction = model(count_matrix, event_type, event_time, volume_matrix)
        
        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood_sparse(model, enc_out, count_matrix, event_time, event_type, volume_matrix)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # print("LOG LIKELIHOOD SPARSE IS OUT")
        
        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, event_volume = map(lambda x: x.to(opt.device), batch)
            
            
            count_matrix = get_count_model_tensor(event_time, event_type, model.bin_size)
            volume_matrix = get_volume_model_tensor(event_time, event_volume, model.bin_size_volume)
            
            """ forward """
            enc_out, prediction = model(count_matrix, event_type, event_time, volume_matrix)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood_sparse(model, enc_out, count_matrix, event_time, event_type, volume_matrix)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-lambda_window', type=int, default=5)
    parser.add_argument('-bin_size', type=float, default=5)
    parser.add_argument('-bin_size_volume', type=float, default=3)
    
    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)
    
    """ prepare model """
    
    model = SparseTransformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,    
        d_v=opt.d_v,
        dropout=opt.dropout,
        lambda_window=opt.lambda_window,
        bin_size=opt.bin_size,
        bin_size_volume=opt.bin_size_volume
    )
    
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
