#!/usr/bin/env bash
set -euo pipefail

echo "=== Mosquitto MQTT Broker Setup ==="

install_macos() {
    if ! command -v brew &>/dev/null; then
        echo "Error: Homebrew not found. Install it from https://brew.sh"
        exit 1
    fi
    if brew list mosquitto &>/dev/null; then
        echo "Mosquitto is already installed."
    else
        echo "Installing Mosquitto via Homebrew..."
        brew install mosquitto
    fi
    echo "Starting Mosquitto service..."
    brew services start mosquitto
}

install_linux() {
    echo "Installing Mosquitto via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y mosquitto mosquitto-clients
    echo "Enabling and starting Mosquitto service..."
    sudo systemctl enable mosquitto
    sudo systemctl start mosquitto
}

case "$(uname -s)" in
    Darwin)
        install_macos
        ;;
    Linux)
        install_linux
        ;;
    *)
        echo "Error: Unsupported OS. Install Mosquitto manually from https://mosquitto.org/download/"
        exit 1
        ;;
esac

# Verify broker is running
echo ""
echo "Verifying broker on localhost:1883..."
sleep 1
if command -v mosquitto_pub &>/dev/null; then
    if mosquitto_pub -h localhost -p 1883 -t "test/setup" -m "ok" -q 0 2>/dev/null; then
        echo "Broker is running and accepting connections."
    else
        echo "Warning: Broker may not be ready yet. Check with: brew services list (macOS) or systemctl status mosquitto (Linux)"
    fi
else
    echo "mosquitto_pub not found, skipping connection test."
fi

echo ""
echo "Done! Broker is configured at localhost:1883 (default)."
