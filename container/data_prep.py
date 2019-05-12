#!/usr/bin/env python3

import pandas as pd
import os
from bokeh.plotting import figure, show, output_file

cwd = os.getcwd()
data_path = os.path.join(cwd, './RR Historical Data.csv')


def get_data(data_path):

    df = pd.read_csv(data_path, parse_dates=True, thousands=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop(df.index[-1:],inplace=True)
    df = df.reset_index(drop=True)
    for i in range(0, len(df)):
        for col in df:
            if df.loc[i, col] == '-':
                df.loc[i, col] = df.loc[i-1, col]
            if col == 'Vol.':
                if df.loc[i, col].endswith('M'):
                    df.loc[i, col] = df.loc[i, col].rstrip('M')
                    df.loc[i, col] = str(float(df.loc[i, col]) * 1e6)
                if df.loc[i, col].endswith('K'):
                    df.loc[i, col] = df.loc[i, col].rstrip('K')
                    df.loc[i, col] = str(float(df.loc[i, col]) * 1e3)
            if col == 'Change %':
                df.loc[i, col] = df.loc[i, col].replace(',', '')
                if df.loc[i, col].endswith('%'):
                    df.loc[i, col] = df.loc[i, col].rstrip('%')
    
    for col in df:
        if not col == 'Date':
            df[col] = pd.to_numeric(df[col])
    return df


def plot_data(df):

    inc = df.Price > df.Open
    dec = df.Open > df.Price
    
    ymin = df['Low'].min()
    ymax = df['High'].max()
    
    w = 12*60*60*1000
    
    TOOLS = 'pan,wheel_zoom,box_zoom,reset,save'
    
    p = figure(x_axis_type='datetime', tools=TOOLS, plot_width=1000, title = 'RR Candlestick', y_range=(ymin, ymax))
    p.xaxis.major_label_orientation = 3.14159/4
    p.grid.grid_line_alpha=0.3
    
    p.segment(df.Date, df.High, df.Date, df.Low, color='black')
    p.vbar(df.Date[inc], w, df.Open[inc], df.Price[inc], fill_color='#D5E1DD', line_color='black')
    p.vbar(df.Date[dec], w, df.Open[dec], df.Price[dec], fill_color='#F2583E', line_color='black')
    
    output_file('rr_candlestick.html', title='LON:RR Candlestick Plot')
    
    show(p)

if __name__=='__main__':
    df = get_data(data_path)
    print(df.tail(100))
    plot_data(df)
