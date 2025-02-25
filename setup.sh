#!/bin/bash

# Check if Python 3.10.16 is installed
PYTHON_VERSION=$(python3 --version 2>&1)

if [[ $PYTHON_VERSION != "Python 3.10.16" ]]; then
    echo "Warning: Python 3.10.16 is not installed. Ensure you are using the correct version."
fi

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
