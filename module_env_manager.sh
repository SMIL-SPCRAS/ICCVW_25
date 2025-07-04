#!/bin/bash

ENV_DIR="/home/daryumin/ICCVW_25/env"
DEFAULT_ENV="Python_PyTorch_GPU_v2_4"

print_usage() {
    echo "Usage:"
    echo "  $0 save <env_name>    - Save the current module environment"
    echo "  $0 restore <env_name> - Restore a saved module environment"
    echo "  $0 delete <env_name>  - Delete a saved module environment"
    echo "  $0 list               - List all saved environments"
    echo
    echo "Notes:"
    echo "  <env_name> can contain only letters, digits, hyphens, and underscores."
    echo "  Saved environments are stored in: $ENV_DIR"
    exit 1
}

if ! command -v module &> /dev/null; then
    source /etc/profile.d/lmod.sh
fi

if [ "$#" -lt 1 ]; then
    echo "No action specified. Restoring default environment: $DEFAULT_ENV"
    if [ ! -f "$ENV_DIR/$DEFAULT_ENV" ]; then
        echo "Error: Default environment '$DEFAULT_ENV' not found in $ENV_DIR."
        exit 1
    fi

    module use /opt/ohpc/pub/modulefiles
    module use /opt/ohpc/private/modulefiles

    cp "$ENV_DIR/$DEFAULT_ENV" "$HOME/.lmod.d/"
    module restore "$DEFAULT_ENV"

    echo "Default environment '$DEFAULT_ENV' successfully restored."
    module list
    exit 0
fi

ACTION="$1"
ENV_NAME="$2"

if [[ "$ACTION" =~ ^(save|restore|delete)$ ]] && [[ ! "$ENV_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: Environment name can only contain letters, digits, hyphens, and underscores."
    exit 1
fi

case "$ACTION" in
    save)
        echo "Saving environment: $ENV_NAME"
        module save "$ENV_NAME"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to save environment with module save."
            exit 1
        fi
        mv "$HOME/.lmod.d/$ENV_NAME" "$ENV_DIR/"
        echo "Environment '$ENV_NAME' successfully saved to $ENV_DIR."
        ;;
    restore)
        if [ ! -f "$ENV_DIR/$ENV_NAME" ]; then
            echo "Error: Environment file '$ENV_DIR/$ENV_NAME' not found."
            exit 1
        fi
        module use /opt/ohpc/pub/modulefiles
        module use /opt/ohpc/private/modulefiles
        echo "Restoring environment: $ENV_NAME"
        cp "$ENV_DIR/$ENV_NAME" "$HOME/.lmod.d/"
        module restore "$ENV_NAME"
        module list
        echo "Environment '$ENV_NAME' successfully restored."
        ;;
    delete)
        if [ ! -f "$ENV_DIR/$ENV_NAME" ]; then
            echo "Error: Environment '$ENV_NAME' not found in $ENV_DIR."
            exit 1
        fi
        rm -f "$ENV_DIR/$ENV_NAME"
        echo "Environment '$ENV_NAME' deleted from $ENV_DIR."
        ;;
    list)
        echo "Available environments in $ENV_DIR:"
        ls "$ENV_DIR"
        ;;
    *)
        echo "Error: Unknown action '$ACTION'."
        print_usage
        ;;
esac