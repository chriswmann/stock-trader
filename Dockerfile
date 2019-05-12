FROM python:3

run apt-get update && apt-get -y install vim
run pip install pytest numpy pandas sklearn xgboost hyperopt \
joblib tensorflow bokeh alpha_vantage argparse plotter matplotlib
run pip install matplotlib2tikz

COPY .vimrc /root
COPY .viminfo /root
COPY .vim/ /root/.vim

WORKDIR /data
