#!/usr/bin/env python3

import arg_parser
import numpy as np
import os
import pandas as pd
import pickle
import re
import tensorflow as tf
import time

import alpha_vantage_key
import arg_parser
import plotter
#import weather

from alpha_vantage.timeseries import TimeSeries
from bokeh.models import LinearAxis
from bokeh.palettes import Spectral11
from bokeh.plotting import figure, show, output_file
from pprint import pprint
from sklearn import preprocessing

config_folder, refresh, plot_inputs, plot_results, test_model, train_model, max_epoch, init_learning_rate, init_eopch, weather_data, ticker = arg_parser.get_args()

cwd = os.getcwd()
if not config_folder and refresh:
    today = time.strftime('%d-%m-%Y')
    config_folder = f'{ticker}_{today}'
elif not config_folder and not refresh:
    all_subdirs = [d for d in os.listdir(cwd) if os.path.isdir(d) and re.match(r'.{1,6}_[0-9]{2}-[0-9]{2}-[0-9]{4}', d)]
    print(all_subdirs)
    config_folder = max(all_subdirs, key=os.path.getmtime)
config_path = os.path.join(cwd, config_folder, f'{ticker}_trader_config.pkl')
data_path = os.path.join(cwd, config_folder, f'{ticker}_data.pkl')
meta_data_path = os.path.join(cwd, config_folder, f'{ticker}_meta_data.pkl')
plot_path = os.path.join(cwd, 'plots')

latest_data_path = os.path.join(cwd, config_folder)
if not os.path.exists(latest_data_path):
    os.makedirs(latest_data_path)

if not os.path.exists(plot_path):
    os.makedirs(plot_path)


def avg(data):
    data = list(data)
    return sum(data)/len(data)

def now_str():
    return time.strftime('%H-%M-%S', time.gmtime())

# RNN model save info
model_name = 'trader_model.dat'
save_model_path = os.path.join(cwd, config_folder, model_name)

def display_output(data, titles, print_data=False):
        
        print(data.shape)
        if print_data:
            print(daax)
        print(f'Shape of data is: {data.shape}')
        for i, c in enumerate(data.T):
            print(data.T[i])
            print(min(data.T[i]))
            print(max(data.T[i]))
            print(f'Min {c}: {min(data.T[i])}')
            print(f'Max {c}: {max(data.T[i])}')


class RNNConfig(object):
    init_epoch = 5
    init_learning_rate = 0.0001
    keep_prob = 0.2
    learning_rate_decay = 0.95
    min_learning_rate = 1e-6
    lstm_size = 8
    max_epoch = 100
    num_layers = 4
    batch_size = 2
    time_steps = 7
    test_percent = 0.2
    # shift is the number of days in the future to try to predict
    shift = 7
    tickers = [ticker]
    tickers = ['RR.L', '^FTSE', '^FTMC', 'BA.L', 'GE', 'UTX']
    # tickers with insufficient data: SNR GEK AIR.DU AAL BRNT.L



class getData(object):

    def __init__(self, tickers, config_path, data_path, meta_data_path, weather_data, refresh_data=False):
        self.tickers = tickers
        self.config_path = config_path
        self.data_path = data_path
        self.meta_data_path = meta_data_path
        self.refresh_data = refresh_data
        self.weather_data = weather_data

    def read_data(self):
        if (not os.path.isfile(self.data_path) and not os.path.isfile(self.meta_data_path)) or self.refresh_data:
            ts = TimeSeries(key=alpha_vantage_key.key, output_format='pandas')
            self.tick_data = {}
            self.meta_data = {}
            for t in self.tickers:
                print(f'Downloading {t} data...')

                td, md = ts.get_daily(symbol=t, outputsize='full')
                if t == 'RR.L':
                    tick_len = td.shape[0]
                else:
                    '''
                    NB need to sort this section out!
                    '''
                    tick_len = 1000
                if len(td) >= tick_len:
                    self.tick_data[t] = {}
                    self.tick_data[t]['data'], self.tick_data[t]['meta'] = td, md
                    old_titles = self.tick_data[t]['data'].columns.values
                    new_titles = [title + f' {t}' for title in old_titles]
                    titles = dict(zip(old_titles, new_titles))
                    self.tick_data[t]['data'].rename(columns = titles, inplace=True)
                    print('Download complete.')
                else:
                    print(f'Discarded {t} due to insufficient historical data.')
                    print(f'RR.L length is {tick_len}, {t} length is {len(td)}.')
            for t in self.tick_data.keys():
                if t == self.tickers[0]:
                    self.data = self.tick_data[t]['data']
                else:
                    self.data = pd.concat([self.data, self.tick_data[t]['data']], axis=1)
                self.meta_data[t] = self.tick_data[t]['meta']

            print(self.data.shape)
            self.data.to_pickle(self.data_path)
            self.meta_data['data_path'] = self.data_path
            with open(self.meta_data_path, 'wb') as f:
                pickle.dump(self.meta_data, f)
        else:
            if not self.refresh_data:
                print('Not refreshing data.')
            with open(self.meta_data_path, 'rb') as f:
                self.meta_data = pickle.load(f)
            self.data_path = self.meta_data['data_path']
            self.data = pd.read_pickle(self.data_path)
        
        # back and forward fill NaNs
        self.data.bfill().ffill()
        self.data = self.data.dropna()
        print(self.data.shape)
        # normalise data
        x = self.data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.data = pd.DataFrame(x_scaled, columns = self.data.columns)
        self.orig_titles = self.data.columns.values
        self.titles = [name.split()[-1] for name in self.orig_titles]
        self.data.reset_index()

        return self.data, self.meta_data, self.titles


class prepareData(object):
    
    def __init__(self, data, time_steps, batch_size, shift, test_percent):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.shift = shift
        self.data = data
        self.data_len = len(self.data)
        self.n_features = self.data.shape[1]
        self.train_percent = 1 - test_percent


        self.batch_len = self.batch_size * self.time_steps
        self.raw_train_len = int(self.data_len * self.train_percent)
        self.n_train_batches = int(self.raw_train_len / self.batch_len)
        self.train_len = int(self.n_train_batches * self.batch_len)
        self.train_data = self.data[:self.train_len]
        
        self.raw_test_len = int(self.data_len - self.train_len)
        self.n_test_batches = int(self.raw_test_len / self.batch_len)
        self.test_len = self.n_test_batches * self.batch_len
        self.test_data = self.data[self.train_len:self.train_len + self.test_len]

    def prep_train_data(self):
        self.train_data = np.reshape(self.train_data, [-1, self.batch_size, self.time_steps, self.n_features])
        self.train_labels = []
        for i, outter_arr in enumerate(self.train_data):
            if i > 0:
                # RR close price is the required label, at the time of this comment
                # it is at column index 3
                self.train_labels.append(outter_arr[:,:,3])
        self.train_data = self.train_data[:-1]
        
    def prep_test_data(self):
        self.test_data = np.reshape(self.test_data, [-1, self.batch_size, self.time_steps, self.n_features])
        self.test_labels = []
        for i, outter_arr in enumerate(self.test_data):
            if i > 0:
                # RR close price is the required label, at the time of this comment
                # it is at column index 3
                self.test_labels.append(outter_arr[:,:,3])
        self.test_data = self.test_data[:-1]

    def output_train_data(self):
        return self.train_data, self.train_labels, self.n_features

    def output_test_data(self):
        return self.test_data, self.test_labels, self.n_features


class linePlot(object):

    def __init__(self, x, y, title):
        self.p = figure(title=title, plot_width=800, plot_height=600)
        self.p.line(x=x, y=y)

    def show_plot(self):
        show(self.p) 


if __name__=='__main__':
    os.system('clear')
   
    print(f'Config folder being used is {config_folder}')
    
    config = RNNConfig()
    tickers = config.tickers
    batch_size = config.batch_size    
    keep_prob = config.keep_prob
    init_epoch = config.init_epoch
    if not init_learning_rate:
        init_learning_rate = config.init_learning_rate
    if not init_epoch:
        init_epoch = config.init_epoch


    learning_rate_decay = config.learning_rate_decay
    min_learning_rate = config.min_learning_rate
    lstm_size = config.lstm_size
    if not max_epoch:
        max_epoch = config.max_epoch
    num_layers = config.num_layers
    shift = config.shift
    test_percent = config.test_percent
    time_steps = config.time_steps

    # TensorBoard writer info
    writer_path = 'writer'
    try:
        writer_num = int(writer.split(os.sep)[1])
    except (NameError, ValueError):
        writer_num = 0

    model_path = ''
    for item in [lstm_size, num_layers, batch_size, time_steps, learning_rate_decay, shift]:
        model_path += f'{item}-'
    model_path = model_path[:-1]
    writer_path = os.path.join(cwd, writer_path, str(writer_num),  model_path)

    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    yes = ['y', 'yes']
    choice = 'y'

    if test_model and refresh:
        choice =input('Are you sure you want to refresh the data for a test run?')

    if not choice.lower() in yes:
        refresh = False

    gd = getData(tickers, config_path, data_path, meta_data_path, weather_data, refresh_data=refresh)
    dat, meta_data, titles = gd.read_data()
    data = dat.values

    if plot_inputs:
        dplot = plotter.Plot(dat)
        dplot.plot_shape()

    #display_output(data, titles, print_data=False)

    data_prepper = prepareData(data, time_steps, batch_size, shift, test_percent)

    data_prepper.prep_train_data()
    x, y, n_features = data_prepper.output_train_data()
    plotter.Plot(x)
    #display_output(x[0], titles)
    assert_error = f'Input length {len(x)} not equal to target {len(y)}'
    assert len(x) == len(y), assert_error
    tf.reset_default_graph()
    lstm_graph = tf.Graph()
    train_cost = []
    train_data = []
    error_data = []
   
    if plot_inputs:
        time.sleep(0.5)
        xplot = plotter.Plot(x)
        xplot.plot_shape()

    with lstm_graph.as_default() as g:

        inputs = tf.placeholder(tf.float32, [batch_size, time_steps, n_features], name='Inputs')
        labels = tf.placeholder(tf.float32, [batch_size, time_steps], name='Labels')
        learning_rate = tf.placeholder(tf.float32, None, name='Learning_Rate')

        def _create_one_cell():
            return tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True, activation=tf.nn.relu)
            if keep_prob < 1.0:
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keeep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
                [_create_one_cell() for _ in range(num_layers)],
                state_is_tuple=True
                ) if config.num_layers > 1 else _create_one_cell()

        val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name='last_lstm_output')

        weight = tf.Variable(tf.truncated_normal([lstm_size, time_steps]), name='W')
        bias = tf.Variable(tf.constant(0.1, shape=[time_steps]), name='B')
        #prediction = tf.matmul(last, weight) + bias
        prediction = tf.add(tf.matmul(last, weight), bias, name='Prediction')

        
        with tf.name_scope('accuracy'):
            cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(prediction, labels))))
            optimiser = tf.train.AdamOptimizer(learning_rate)
            minimize = optimiser.minimize(cost)
        if train_model:
            with tf.Session(graph=lstm_graph) as train_sess:
                tf.global_variables_initializer().run()
                writer = tf.summary.FileWriter(writer_path)
                writer.add_graph(train_sess.graph)

                learning_rates_to_use = [
                        init_learning_rate * (
                            learning_rate_decay ** max(float(i + 1 - init_epoch), 0)
                            ) for i in range(max_epoch)]
                
                for ind, item in enumerate(learning_rates_to_use):
                    if item < min_learning_rate:
                        learning_rates_to_use[ind] = min_learning_rate
                if plot_inputs:
                    plot = True
                else:
                    plot = False
                xlen = len(x)
                for epoch_step in range(max_epoch):

                    current_lr = learning_rates_to_use[epoch_step]
                
                    for i, batch_x in enumerate(x):
                        if plot:
                            bx_plot = plotter.Plot(batch_x)
                            bx_plot.plot_shape()
                            plot = False
                        batch_y = y[i]
                        batch_x.reshape(batch_size, time_steps, n_features)
                        '''
                        if (xlen - i) < 100:
                            batch_x = np.multiply(batch_x, 100)

                        '''
                        train_data_feed = {
                                inputs: batch_x,
                                labels: batch_y,
                                learning_rate: current_lr
                                }
                        cost_output, train_output = train_sess.run([[cost, minimize], [prediction, labels]], train_data_feed)
                        if epoch_step == max_epoch-1:
                            train_data.append(train_output)
                    train_cost.append(cost_output[0])
                    print(f'Epoch: {epoch_step + 1}')
                    print(f'Current learning rate: {current_lr:.8f}')
                    print(f'End of epoch training cost: {train_cost[-1]:.8f}')
                    error = avg(train_output[0][-1]) - avg(train_output[1][-1])
                    error_data.append(error)
                    print(f'End of epoch error: {error:.8f}')
                saver = tf.train.Saver()
                saver.save(train_sess, save_model_path, global_step=max_epoch)
            if plot_results:
                x = range(len(train_cost))
                assert_error = f'Abscissa length {len(x)} not equal to ordinate (train_cost) length {len(train_cost)}'
                assert len(x) == len(train_cost), assert_error
                assert_error_data = f'Abscissa length {len(x)} not equal to ordinate (error_data) length {len(error_data)}'
                assert len(x) == len(error_data), assert_error_data
                p1 = figure(title='Cost and Error versus Epoch',
                       plot_width=600,
                       plot_height=400,
                       y_axis_label='Cost')

                p1.line(x,
                        train_cost,
                        legend='Cost',
                        color='firebrick',
                        line_width=1)
                p1.line(x,
                        error_data,
                        #y_range_name='Error',
                        legend='Error',
                        color='navy',
                        line_width=1)
                p1.legend.location = 'top_center'
                #p1.add_layout(LinearAxis(y_range_name='Error', axis_label='Error'), 'right')
                now = now_str()
                output_file(os.path.join(plot_path, f'cost_error_{now}.html'))
                show(p1)
                p2 = figure(title='Prediction vs Actual', plot_width=600, plot_height=400)
                x = []
                yp = []
                yl = []
                for i, r in enumerate(train_data):
                    x.append(i)
                    yp.append(r[0][0][-1])
                    yl.append(r[1][0][-1])
                p2.line(x , yp, legend='Prediction', color='firebrick', line_width=1)
                p2.line(x , yl, legend='Actual', color='navy', line_width=1)
                p2.legend.location = 'top_left'
                output_file(os.path.join(plot_path, f'pred_act_{now}.html'))
                show(p2)

    if test_model:
        with tf.Session() as test_sess:
            graph = tf.Graph()
            ckpt = tf.train.get_checkpoint_state(latest_data_path)
            meta_file_list = [os.path.join(latest_data_path, f) for f in os.listdir(latest_data_path) if f.endswith('meta')]
            meta_file = max(meta_file_list, key=os.path.getmtime)
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(test_sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()

            tf.global_variables_initializer().run()
            data_prepper.prep_test_data()
            x, y , n_features = data_prepper.output_test_data()
            assert_error = f'Abscissa length {len(x)} not equal to ordinate (train_cost) length {len(y)}'
            assert len(x) == len(y), assert_error
            batch_data = x
            inputs = graph.get_tensor_by_name('Inputs:0')
            labels = graph.get_tensor_by_name('Labels:0')
            weight = graph.get_tensor_by_name('W:0')
            bias = graph.get_tensor_by_name('B:0')
            prediction = graph.get_tensor_by_name('Prediction:0')
            test_data = [] 
            for i, batch_x in enumerate(x):
                batch_y = y[i]
                batch_x.reshape(batch_size, time_steps, n_features)
                test_data_feed = {
                        inputs: batch_x,
                        labels: batch_y,
                        }
                test_output = test_sess.run([prediction, labels], test_data_feed)
                test_data.append(test_output)
            p2 = figure(title='Pred vs Label', plot_width=600, plot_height=400)
            x = []
            yp = []
            yl = []
            for i, r in enumerate(test_data):
                x.append(i)
                yp.append(r[0][0][-1])
                yl.append(r[1][0][-1])
            p2.line(x , yp, legend='Prediction', color='firebrick', line_width=1)
            p2.line(x , yl, legend='Actual', color='navy', line_width=1)
            now = now_str()
            output_file(os.path.join(plot_path, f'pred_act_{now}.html'))
            show(p2)
