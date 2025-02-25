# THIS SETUP ONLY WORKS IF YOU HAVE PYTHON 3.10 INSTALLED AND IF YOU ARE USING MAC OS/LINUX

#!/bin/bash

# Try to create the virtual environment with Python 3.10
python3.10 -m venv venv || { echo "Error: Python 3.10 is not installed. Install it and try again."; exit 1; }

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
