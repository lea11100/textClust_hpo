if command -v module >/dev/null 2>&1; then
    module load lang/Python/3.11.3-GCCcore-12.3.0
else
    echo "Environment Modules is not available; skipping module load."
fi
echo "Using Python $(python --version) from $(which python)"

# Check if the virtual environment already exists
if [ ! -d ".textClust_hpo_venv" ]; then
    # Create a new virtual environment
    python -m venv .textClust_hpo_venv
    echo "Created virtual environment in .textClust_hpo_venv"
else
    echo "Virtual environment already exists in .textClust_hpo_venv"
fi

# Activate the virtual environment
source .textClust_hpo_venv/bin/activate
echo "Activated virtual environment in .textClust_hpo_venv"

# Install required packages
echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt
echo "Installed required packages."

echo "Install local packages..."
pip install -e ./ext_lib/textClust
pip install -e ./ext_lib/mosmac

echo "Installed local packages."

echo "Try to install pygmo..."
pip install pygmo

echo "Setup complete. You can now use the virtual environment."
