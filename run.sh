#!/bin/bash

# Function to create a directory if it doesn't exist
create_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Function to create and activate a Python virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        if [ "$(uname)" == "Darwin" ]; then
            # For macOS
            python3 -m venv venv
        else
            # For other platforms (e.g., Windows)
            python -m venv venv
        fi
    fi

    if [ "$(uname)" == "Darwin" ]; then
        # For macOS
        source venv/bin/activate
    else
        # For other platforms (e.g., Windows)
        source venv/Scripts/activate
    fi

#     echo "Installing dependencies from requirements.txt..."
#     pip install -r requirements.txt
}

# Call the setup_venv function to create/activate the virtual environment and install dependencies
setup_venv

# Rest of your script
timestamp=$(date +"%d-%m-%YT%H-%M-%S")
create_directory "results"
create_directory "results/$timestamp"

echo "Running train_mappo.py..."
if [ "$(uname)" == "Darwin" ]; then
    # For macOS
    python3 train_mappo_.py "$timestamp"
else
    # For other platforms (e.g., Windows)
    python train_mappo_.py "$timestamp"
fi

echo "Running test_mappo.py..."
if [ "$(uname)" == "Darwin" ]; then
    # For macOS
    python3 test_mappo_.py "$timestamp"
else
    # For other platforms (e.g., Windows)
    python test_mappo_.py "$timestamp"
fi

echo "Running rescheduling.py..."
if [ "$(uname)" == "Darwin" ]; then
    # For macOS
    python3 rescheduling.py "$timestamp"
else
    # For other platforms (e.g., Windows)
    python rescheduling.py "$timestamp"
fi

# Open TensorBoard
echo "Opening TensorBoard..."
tensorboard --logdir=./logs --port=6006 &

# Wait for TensorBoard to start
sleep 5

# Open TensorBoard in the default browser
if [ "$(uname)" == "Darwin" ]; then
    open http://localhost:6006
elif [ "$(uname)" == "Windows" ]; then
    start http://localhost:6006
else
    echo "Unable to automatically open browser. Please navigate to http://localhost:6006 to view TensorBoard."
fi

echo "Script completed. TensorBoard is running in the background."

# Deactivate the virtual environment
deactivate

echo "Scripts executed successfully and TensorBoard is running on port 6000."