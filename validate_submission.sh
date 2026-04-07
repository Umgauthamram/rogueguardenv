
log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "PASSED -- $1"; }
fail() { log "FAILED -- $1"; }
stop_at() {
  printf "\nValidation stopped at %s. Fix the above before continuing.\n" "$1"
  exit 1
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

PING_URL="${PING_URL%/}"

log "Step 1/3: Pinging local server ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 10 || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "Server is live and responds to /reset"
else
  fail "Server /reset returned HTTP $HTTP_CODE (expected 200)"
  stop_at "Step 1"
fi

log "Step 2/3: Checking OpenEnv spec compliance ..."
if [ ! -f "$REPO_DIR/openenv.yaml" ]; then
  fail "openenv.yaml not found"
  stop_at "Step 2"
fi
pass "openenv.yaml exists"

log "Step 3/3: Verifying inference script ..."
if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "inference.py not found"
  stop_at "Step 3"
fi
pass "inference.py exists"

printf "\nSUBMISSION READY: All checks passed.\n"
