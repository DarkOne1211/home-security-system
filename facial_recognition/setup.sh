#!/bin/bash
apt-get install libatlas-base-dev
apt-get install libjasper-dev
apt-get install libqtgui4
apt-get install python3-pyqt5
apt-get install libqt4-test
python3 -m venv ./
pip3 install -r requirements.txt