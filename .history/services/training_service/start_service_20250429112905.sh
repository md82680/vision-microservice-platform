#!/bin/bash

# Get the environment name and port from arguments
ENV_NAME=$1
PORT=$2

# Create a temporary script to run in a clean environment
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $1
echo "Starting Training Service on port $2"
uvicorn src.app:app --host 0.0.0.0 --port $2
EOF

# Make the script executable and run it
chmod +x "$TEMP_SCRIPT"
bash "$TEMP_SCRIPT" "$ENV_NAME" "$PORT" 