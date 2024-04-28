'''
Author: WANG CHENG
Date: 2024-04-29 01:38:50
LastEditTime: 2024-04-29 01:41:15
'''
import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser(description='occ project parameters')
    parser.add_argument('--data_path', type=str, default='data/occ_data.csv', help='path to the data file')
    parser.add_argument('--model_path', type=str, default='models/model.pth', help='path to save the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to run the model on')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for training')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension for the LSTM layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for the LSTM layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate for the LSTM layer')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes for the classification task')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--log_interval', type=int, default=10, help='interval for logging training status')
    parser.add_argument('--val_interval', type=int, default=1, help='interval for validation')
    parser.add_argument('--save_interval', type=int, default=1, help='interval for saving the trained model')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--max_num_epochs', type=int, default=100, help='maximum number of training epochs')
    parser.add_argument('--max_num_epochs_without_improvement', type=int, default=10, help='maximum number of epochs without improvement')
    parser.add_argument('--min_num_epochs_without_improvement', type=int, default=5, help='minimum number of epochs without improvement')
    parser.add_argument('--min_num_epochs_without_improvement_threshold', type=float, default=0.01, help='minimum number of epochs without improvement threshold')
    parser.add_argument('--min_num_epochs_without_improvement_patience', type=int, default=5, help='minimum number of epochs without improvement patience')
    parser.add_argument('--min_num_epochs_without_improvement_patience_threshold', type=float, default=0.01, help='minimum number of epochs without improvement patience threshold')
    
    return parser 
    
if __name__ == '__main__':
    get_args()