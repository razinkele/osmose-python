#!/usr/bin/env bash
# deploy.sh — Deploy osmose-python to the local Shiny server
#
# Creates a symlink in /srv/shiny-server/osmose pointing to this project,
# installs missing Python dependencies, updates shiny-server.conf, and
# restarts the Shiny server.
#
# Usage:  sudo bash deploy.sh
#         sudo bash deploy.sh --uninstall

set -euo pipefail

APP_NAME="osmose"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
SHINY_ROOT="/srv/shiny-server"
SHINY_CONF="/etc/shiny-server/shiny-server.conf"
SHINY_PYTHON="/opt/micromamba/envs/shiny/bin/python3"
SHINY_PIP="/opt/micromamba/envs/shiny/bin/pip"
LINK_PATH="${SHINY_ROOT}/${APP_NAME}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*" >&2; }

# --- Uninstall mode ---
if [[ "${1:-}" == "--uninstall" ]]; then
    info "Uninstalling ${APP_NAME} from Shiny server..."

    if [[ -L "$LINK_PATH" ]]; then
        rm "$LINK_PATH"
        info "Removed symlink ${LINK_PATH}"
    else
        warn "No symlink at ${LINK_PATH}"
    fi

    if grep -q "location /${APP_NAME}" "$SHINY_CONF" 2>/dev/null; then
        sed -i "/# --- osmose-python start ---/,/# --- osmose-python end ---/d" "$SHINY_CONF"
        info "Removed ${APP_NAME} location block from ${SHINY_CONF}"
    fi

    systemctl restart shiny-server 2>/dev/null && info "Shiny server restarted" || warn "Could not restart shiny-server"
    info "Uninstall complete."
    exit 0
fi

# --- Pre-flight checks ---
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)."
    exit 1
fi

if [[ ! -f "${SOURCE_DIR}/app.py" ]]; then
    error "app.py not found in ${SOURCE_DIR}. Run this script from the project root."
    exit 1
fi

if [[ ! -d "$SHINY_ROOT" ]]; then
    error "Shiny server directory ${SHINY_ROOT} not found."
    exit 1
fi

if [[ ! -f "$SHINY_PYTHON" ]]; then
    error "Shiny Python not found at ${SHINY_PYTHON}."
    exit 1
fi

# --- Step 1: Create symlink ---
if [[ -L "$LINK_PATH" ]]; then
    current_target="$(readlink "$LINK_PATH")"
    if [[ "$current_target" == "$SOURCE_DIR" ]]; then
        info "Symlink already exists: ${LINK_PATH} -> ${SOURCE_DIR}"
    else
        warn "Symlink exists but points to ${current_target}. Updating..."
        rm "$LINK_PATH"
        ln -s "$SOURCE_DIR" "$LINK_PATH"
        info "Updated symlink: ${LINK_PATH} -> ${SOURCE_DIR}"
    fi
elif [[ -e "$LINK_PATH" ]]; then
    error "${LINK_PATH} exists and is not a symlink. Remove it manually."
    exit 1
else
    ln -s "$SOURCE_DIR" "$LINK_PATH"
    info "Created symlink: ${LINK_PATH} -> ${SOURCE_DIR}"
fi

# Ensure shiny user can read the source directory
chown -h shiny:shiny "$LINK_PATH" 2>/dev/null || true

# --- Step 2: Install missing Python dependencies ---
MISSING_PKGS=()
for pkg in pymoo SALib; do
    if ! "$SHINY_PIP" show "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing packages: ${MISSING_PKGS[*]}"
    "$SHINY_PIP" install "${MISSING_PKGS[@]}" --quiet
    info "Packages installed."
else
    info "All Python dependencies already installed."
fi

# --- Step 3: Update shiny-server.conf ---
if grep -q "location /${APP_NAME}" "$SHINY_CONF" 2>/dev/null; then
    info "Shiny server config already has /${APP_NAME} location block."
else
    info "Adding /${APP_NAME} location block to ${SHINY_CONF}..."
    # Insert before the closing brace of the server block
    OSMOSE_BLOCK=$(cat <<'CONF'

  # --- osmose-python start ---
  location /osmose {
    app_dir /srv/shiny-server/osmose;
    python /opt/micromamba/envs/shiny/bin/python3;
    log_dir /var/log/shiny-server;
  }
  # --- osmose-python end ---
CONF
)
    # Find the last closing brace and insert before it
    # Create a backup first
    cp "$SHINY_CONF" "${SHINY_CONF}.bak"
    # Use awk to insert before the final closing brace
    awk -v block="$OSMOSE_BLOCK" '
        /^}/ && !done { print block; done=1 }
        { print }
    ' "${SHINY_CONF}.bak" > "$SHINY_CONF"
    info "Config updated (backup at ${SHINY_CONF}.bak)."
fi

# --- Step 4: Restart Shiny server ---
if systemctl restart shiny-server 2>/dev/null; then
    info "Shiny server restarted."
else
    warn "Could not restart shiny-server via systemctl. Try manually."
fi

# --- Summary ---
echo ""
info "Deployment complete!"
echo "  App URL:    http://localhost:3838/${APP_NAME}/"
echo "  Source:     ${SOURCE_DIR}"
echo "  Symlink:    ${LINK_PATH}"
echo "  Config:     ${SHINY_CONF}"
echo ""
echo "  To uninstall:  sudo bash ${SOURCE_DIR}/deploy.sh --uninstall"
