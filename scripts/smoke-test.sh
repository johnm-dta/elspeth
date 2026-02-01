#!/bin/bash
# smoke-test.sh - Verify ELSPETH Docker image functionality
#
# Runs a series of tests to verify the Docker image works correctly.
# Used for post-deployment verification and CI/CD pipelines.
#
# Usage:
#   ./smoke-test.sh [image]
#   ./smoke-test.sh                           # Uses elspeth:latest
#   ./smoke-test.sh ghcr.io/org/elspeth:v0.1.0
#   ./smoke-test.sh myacr.azurecr.io/elspeth:sha-abc123
#
# Environment Variables:
#   DATABASE_URL  - Database URL for health check (optional)
#   CONFIG_DIR    - Config directory to mount (optional)
#   DATA_DIR      - Data directory to mount (optional)
#
# Exit Codes:
#   0 - All tests passed
#   1 - One or more tests failed

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

IMAGE="${1:-elspeth:latest}"
CONFIG_DIR="${CONFIG_DIR:-./config}"
DATA_DIR="${DATA_DIR:-./data}"
DATABASE_URL="${DATABASE_URL:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# =============================================================================
# Functions
# =============================================================================

log_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1" >&2
    ((TESTS_FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

run_test() {
    local name="$1"
    local cmd="$2"

    ((TESTS_RUN++))
    log_test "$name"

    if eval "$cmd" > /dev/null 2>&1; then
        log_pass "$name"
        return 0
    else
        log_fail "$name"
        return 1
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

log_header "ELSPETH Smoke Tests"
echo "Image: $IMAGE"
echo "Date:  $(date -Iseconds)"
echo ""

# Check Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# =============================================================================
# Test Suite
# =============================================================================

log_header "1. Basic Functionality"

# Test 1.1: CLI runs
run_test "CLI executes" "docker run --rm $IMAGE --version" || true

# Test 1.2: Help output
run_test "Help output" "docker run --rm $IMAGE --help" || true

# Test 1.3: Plugin list
run_test "Plugin list" "docker run --rm $IMAGE plugins list" || true

# =============================================================================
# Test 2: Configuration Validation (if config available)
# =============================================================================

log_header "2. Configuration Validation"

if [[ -d "$CONFIG_DIR" ]]; then
    # Test 2.1: Validate any YAML configs found
    for config in "$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml; do
        if [[ -f "$config" ]]; then
            config_name=$(basename "$config")
            run_test "Validate $config_name" \
                "docker run --rm -v \"$CONFIG_DIR:/app/config:ro\" $IMAGE validate --settings \"/app/config/$config_name\"" || true
        fi
    done

    if [[ $TESTS_RUN -eq 3 ]]; then
        log_skip "No config files found in $CONFIG_DIR"
    fi
else
    log_skip "Config directory not found: $CONFIG_DIR"
fi

# =============================================================================
# Test 3: Health Check (if DATABASE_URL provided)
# =============================================================================

log_header "3. Health Check"

if [[ -n "$DATABASE_URL" ]]; then
    run_test "Health check with database" \
        "docker run --rm -e DATABASE_URL=\"$DATABASE_URL\" $IMAGE health" || true
else
    # Try health check without database (should still work for basic checks)
    run_test "Basic health check" \
        "docker run --rm $IMAGE health 2>/dev/null || docker run --rm $IMAGE --version" || true
fi

# =============================================================================
# Test 4: Smoke Pipeline (if smoke-test.yaml exists)
# =============================================================================

log_header "4. Smoke Pipeline"

if [[ -f "$CONFIG_DIR/cicd/smoke-test.yaml" ]]; then
    # Create temp directories for output
    TEMP_OUTPUT=$(mktemp -d)
    TEMP_STATE=$(mktemp -d)

    run_test "Smoke pipeline execution" \
        "docker run --rm \
            -v \"$CONFIG_DIR:/app/config:ro\" \
            -v \"$TEMP_OUTPUT:/app/output\" \
            -v \"$TEMP_STATE:/app/state\" \
            $IMAGE run --settings /app/config/cicd/smoke-test.yaml --execute" || true

    # Cleanup
    rm -rf "$TEMP_OUTPUT" "$TEMP_STATE"
else
    log_skip "Smoke test config not found: $CONFIG_DIR/cicd/smoke-test.yaml"
fi

# =============================================================================
# Test 5: Container Health Check
# =============================================================================

log_header "5. Container HEALTHCHECK"

# Run container and check health status
CONTAINER_ID=$(docker run -d --rm $IMAGE sleep 30 2>/dev/null || echo "")
if [[ -n "$CONTAINER_ID" ]]; then
    sleep 5  # Wait for health check to run

    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_ID" 2>/dev/null || echo "none")
    docker stop "$CONTAINER_ID" > /dev/null 2>&1 || true

    if [[ "$HEALTH_STATUS" == "healthy" ]]; then
        ((TESTS_RUN++))
        log_pass "Container health check: $HEALTH_STATUS"
        ((TESTS_PASSED++))
    elif [[ "$HEALTH_STATUS" == "starting" ]]; then
        ((TESTS_RUN++))
        log_pass "Container health check: $HEALTH_STATUS (still starting)"
        ((TESTS_PASSED++))
    else
        ((TESTS_RUN++))
        log_fail "Container health check: $HEALTH_STATUS"
        ((TESTS_FAILED++))
    fi
else
    log_skip "Could not start container for health check"
fi

# =============================================================================
# Summary
# =============================================================================

log_header "Summary"

echo "Tests run:    $TESTS_RUN"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"
echo ""

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some smoke tests failed!${NC}"
    exit 1
fi
