#!/usr/bin/env bash
# BrainDB installation script — builds CLI, Server, and Python package.
#
# Usage:
#   ./install.sh              # Build and install everything
#   ./install.sh cli          # Build CLI only
#   ./install.sh server       # Build Server only
#   ./install.sh python       # Build Python wheel only
#   ./install.sh release      # Build optimized release binaries

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/target"
BIN_DIR="$HOME/.local/bin"

build_cli() {
    echo "🔧 Building braindb-cli..."
    cargo build --features cli --no-default-features "$@"
    mkdir -p "$BIN_DIR"
    cp "$BUILD_DIR/debug/braindb-cli" "$BIN_DIR/braindb" 2>/dev/null \
       || cp "$BUILD_DIR/debug/braindb-cli.exe" "$BIN_DIR/braindb-cli.exe" 2>/dev/null \
       || true
    echo "✅ braindb-cli installed to $BIN_DIR"
}

build_server() {
    echo "🔧 Building braindb-server..."
    cargo build --features server --no-default-features "$@"
    mkdir -p "$BIN_DIR"
    cp "$BUILD_DIR/debug/braindb-server" "$BIN_DIR/" 2>/dev/null \
       || cp "$BUILD_DIR/debug/braindb-server.exe" "$BIN_DIR/" 2>/dev/null \
       || true
    echo "✅ braindb-server installed to $BIN_DIR"
}

build_python() {
    echo "🔧 Building braindb Python package..."
    pip install maturin
    maturin develop --features pyo3-extension-module --no-default-features
    echo "✅ braindb Python package installed (pip install braindb)"
}

build_release() {
    echo "🔧 Building release binaries..."
    cargo build --release --features cli,server --no-default-features
    mkdir -p "$BIN_DIR"
    cp "$BUILD_DIR/release/braindb-cli" "$BIN_DIR/" 2>/dev/null || true
    cp "$BUILD_DIR/release/braindb-server" "$BIN_DIR/" 2>/dev/null || true
    echo "✅ Release binaries installed to $BIN_DIR"
}

case "${1:-all}" in
    cli)     build_cli ;;
    server)  build_server ;;
    python)  build_python ;;
    release) build_release ;;
    all)
        build_cli "$@"
        build_server "$@"
        echo ""
        echo "🎉 BrainDB installed! Commands:"
        echo "   braindb-cli build -o net.braindb -n 100"
        echo "   braindb-cli info net.braindb"
        echo "   braindb-cli run net.braindb -d 100 --stimulus '0:30'"
        echo "   braindb-cli query net.braindb downstream 0"
        echo "   braindb-server  # → http://localhost:3000"
        ;;
    *)
        echo "Usage: $0 [cli|server|python|release|all]"
        exit 1
        ;;
esac
