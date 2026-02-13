# Threshold Gate Container Example

The [`threshold_gate`](../threshold_gate/) example packaged for Docker deployment.

## What This Shows

The same pipeline as `threshold_gate` but with container-appropriate paths (`/app/pipeline/...`) for running inside the ELSPETH Docker image.

## Running

```bash
# Build the ELSPETH image
docker build -t elspeth .

# Run the containerised pipeline
docker run -v $(pwd)/examples/threshold_gate_container:/app/pipeline elspeth \
  run --settings /app/pipeline/settings.yaml --execute
```

## Output

- `/app/pipeline/output/high_values.csv` — Rows where `amount > 1000`
- `/app/pipeline/output/normal.csv` — Rows where `amount <= 1000`

(Mapped to `examples/threshold_gate_container/output/` on the host via the volume mount.)

## Key Concepts

- **Container paths**: Uses `/app/pipeline/` instead of relative `examples/` paths
- **Volume mounting**: Input/output/audit data lives on the host, mounted into the container
- **Same pipeline logic**: Identical gate condition and routing as the non-container variant
