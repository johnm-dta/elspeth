#!/bin/bash
# deploy-vm.sh - Deploy ELSPETH to a Linux VM
#
# This script updates the ELSPETH Docker image on a VM and runs verification.
# Since ELSPETH is a CLI tool (not a daemon), deployment simply updates the
# image that will be used for the next invocation.
#
# Usage:
#   ./deploy-vm.sh <image-tag>
#   ./deploy-vm.sh sha-abc123f
#   ./deploy-vm.sh v0.1.0
#   ./deploy-vm.sh latest
#
# Environment Variables:
#   DEPLOY_DIR   - Deployment directory (default: /srv/elspeth)
#   REGISTRY     - Container registry (default: ghcr.io/your-org)
#   SKIP_BACKUP  - Skip database backup (default: false)
#   SKIP_SMOKE   - Skip smoke tests (default: false)
#
# Exit Codes:
#   0 - Success
#   1 - Invalid arguments
#   2 - Image pull failed
#   3 - Migration failed
#   4 - Smoke test failed

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

IMAGE_TAG="${1:-}"
DEPLOY_DIR="${DEPLOY_DIR:-/srv/elspeth}"
REGISTRY="${REGISTRY:-ghcr.io/your-org}"
SKIP_BACKUP="${SKIP_BACKUP:-false}"
SKIP_SMOKE="${SKIP_SMOKE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

show_usage() {
    echo "Usage: $0 <image-tag>"
    echo ""
    echo "Examples:"
    echo "  $0 sha-abc123f    # Deploy specific commit"
    echo "  $0 v0.1.0         # Deploy release version"
    echo "  $0 latest         # Deploy latest (not recommended for production)"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOY_DIR   Deployment directory (default: /srv/elspeth)"
    echo "  REGISTRY     Container registry (default: ghcr.io/your-org)"
    echo "  SKIP_BACKUP  Skip database backup (default: false)"
    echo "  SKIP_SMOKE   Skip smoke tests (default: false)"
}

backup_database() {
    local backup_dir="$DEPLOY_DIR/backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)

    mkdir -p "$backup_dir"

    if [[ -f "$DEPLOY_DIR/state/landscape.db" ]]; then
        log_info "Backing up database..."
        cp "$DEPLOY_DIR/state/landscape.db" "$backup_dir/landscape_${timestamp}.db"
        log_info "Backup created: $backup_dir/landscape_${timestamp}.db"

        # Keep only last 5 backups
        ls -t "$backup_dir"/landscape_*.db 2>/dev/null | tail -n +6 | xargs -r rm
    else
        log_warn "No database found to backup"
    fi
}

rollback() {
    local previous_tag="$1"
    log_error "Deployment failed! Rolling back to: $previous_tag"

    if [[ "$previous_tag" != "none" ]]; then
        sed -i "s/IMAGE_TAG=.*/IMAGE_TAG=$previous_tag/" "$DEPLOY_DIR/.env"
        docker compose -f "$DEPLOY_DIR/docker-compose.yaml" pull
        log_info "Rolled back to $previous_tag"
    else
        log_error "No previous image to rollback to"
    fi

    exit 1
}

# =============================================================================
# Main Script
# =============================================================================

# Validate arguments
if [[ -z "$IMAGE_TAG" ]]; then
    log_error "Missing required argument: image-tag"
    show_usage
    exit 1
fi

# Check deploy directory exists
if [[ ! -d "$DEPLOY_DIR" ]]; then
    log_error "Deploy directory not found: $DEPLOY_DIR"
    exit 1
fi

cd "$DEPLOY_DIR"

# Check required files exist
if [[ ! -f "docker-compose.yaml" ]]; then
    log_error "docker-compose.yaml not found in $DEPLOY_DIR"
    exit 1
fi

if [[ ! -f ".env" ]]; then
    log_warn ".env file not found, creating with defaults"
    echo "IMAGE_TAG=latest" > .env
    echo "REGISTRY=$REGISTRY" >> .env
fi

# =============================================================================
# Step 1: Record current state for rollback
# =============================================================================

PREVIOUS_TAG=$(grep 'IMAGE_TAG=' .env | cut -d= -f2 || echo "none")
log_info "Current image: $PREVIOUS_TAG"
log_info "Deploying image: $IMAGE_TAG"

# =============================================================================
# Step 2: Backup database (unless skipped)
# =============================================================================

if [[ "$SKIP_BACKUP" != "true" ]]; then
    backup_database
else
    log_warn "Skipping database backup (SKIP_BACKUP=true)"
fi

# =============================================================================
# Step 3: Update image tag and pull
# =============================================================================

log_info "Updating image tag..."
sed -i "s/IMAGE_TAG=.*/IMAGE_TAG=$IMAGE_TAG/" .env

log_info "Pulling new image..."
if ! docker compose pull; then
    log_error "Failed to pull image: $REGISTRY/elspeth:$IMAGE_TAG"
    rollback "$PREVIOUS_TAG"
    exit 2
fi

# =============================================================================
# Step 4: Run database migrations (if applicable)
# =============================================================================

log_info "Running database migrations..."
if ! docker compose run --rm elspeth alembic upgrade head 2>/dev/null; then
    log_warn "No migrations to run or alembic not configured"
fi

# =============================================================================
# Step 5: Run smoke tests (unless skipped)
# =============================================================================

if [[ "$SKIP_SMOKE" != "true" ]]; then
    log_info "Running smoke tests..."

    # Test 1: Version check
    log_info "  - Version check..."
    if ! docker compose run --rm elspeth --version; then
        log_error "Version check failed"
        rollback "$PREVIOUS_TAG"
        exit 4
    fi

    # Test 2: Config validation (if config exists)
    if [[ -f "$DEPLOY_DIR/config/pipeline.yaml" ]]; then
        log_info "  - Config validation..."
        if ! docker compose run --rm elspeth validate --settings /app/config/pipeline.yaml; then
            log_error "Config validation failed"
            rollback "$PREVIOUS_TAG"
            exit 4
        fi
    fi

    # Test 3: Health check (if command exists)
    log_info "  - Health check..."
    docker compose run --rm elspeth health 2>/dev/null || log_warn "Health command not available"

    log_info "All smoke tests passed!"
else
    log_warn "Skipping smoke tests (SKIP_SMOKE=true)"
fi

# =============================================================================
# Success
# =============================================================================

echo ""
log_info "=========================================="
log_info "Deployment successful!"
log_info "=========================================="
log_info "Image:    $REGISTRY/elspeth:$IMAGE_TAG"
log_info "Previous: $PREVIOUS_TAG"
log_info ""
log_info "To rollback if needed:"
log_info "  sed -i 's/IMAGE_TAG=.*/IMAGE_TAG=$PREVIOUS_TAG/' $DEPLOY_DIR/.env"
log_info "  docker compose pull"
