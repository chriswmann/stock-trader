#!/usr/bin/env python3

import argparse

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '--weather_data',
            help='include weather data in model',
            action='store_true',
            )
    
    parser.add_argument(
            '--config_folder',
            type=str,
            help='provide a path to the desired config folder',
            action='store',
            )
    
    parser.add_argument(
            '--init_learning_rate',
            type=float,
            help='specify the initial learning rate',
            action='store',
            )

    parser.add_argument(
            '--init_epoch',
            type=int,
            help='specify the epoch at which to start reducing the learning rate',
            action='store',
            )

    parser.add_argument(
            '--max_epoch',
            type=int,
            help='specify the maximum epoch to run to',
            action='store',
            )
    
    parser.add_argument(
            '--refresh_data',
            help='refresh the ticker data',
            action='store_true',
            )
    
    parser.add_argument(
            '--train_model',
            help='train the model',
            action='store_true',
            )
    
    parser.add_argument(
            '--test_model',
            help='test the model',
            action='store_true',
            )
    
    parser.add_argument(
            '--validate',
            help='validate the model',
            action='store_true',
            )
    
    parser.add_argument(
            '--plot_inputs',
            help='produce bokeh plots of inputs and other working tensors',
            action='store_true',
            )
    
    parser.add_argument(
            '--plot_results',
            help='produce bokeh plots of results',
            action='store_true',
            )
    
    parser.add_argument(
            '--ticker',
            type=str,
            help='specify a single ticker to analyse',
            action='store',
            )
    
    args = parser.parse_args()

    if args.config_folder:
        config_folder = args.config_folder
    else:
        config_folder = False
    
    if args.refresh_data:
        refresh = True
    else:
        refresh = False
    
    if args.plot_inputs:
        plot_inputs = True
    else:
        plot_inputs = False
    
    if args.plot_results:
        plot_results = True
    else:
        plot_results = False
    
    if args.test_model:
        test_model = True
    else:
        test_model = False
    
    if args.train_model:
        train_model = True
    else:
        train_model = False

    if args.max_epoch:
        max_epoch = args.max_epoch
    else:
        max_epoch = False

    if args.max_epoch:
        max_epoch = args.max_epoch
    else:
        max_epoch = False

    if args.init_learning_rate:
        init_learning_rate = args.init_learning_rate
    else:
        init_learning_rate = False

    if args.init_epoch:
        init_epoch = args.init_epoch
    else:
        init_epoch = False
    
    if args.weather_data:
        weather_data = args.weather_data
    else:
        weather_data = False

    if args.ticker:
        ticker = args.ticker
    else:
        ticker = False

    return config_folder, refresh, plot_inputs, plot_results, test_model, train_model, max_epoch, init_learning_rate, init_epoch, weather_data, ticker


if __name__ == '__main__':
   get_args()
