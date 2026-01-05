# Rebuild and Serve

Full rebuild of the fish simulation: export checkpoint to ONNX, rebuild WASM, and start the dev server.

## Steps

1. Export the latest checkpoint to ONNX:
```bash
cd $ARGUMENTS || cd /Users/rtty/enten
python3 -c "
import sys; sys.path.insert(0, '.')
from training.models.export import export_to_onnx
export_to_onnx('checkpoints/policy_final.pt', 'inference/model.onnx')
"
```

2. Rebuild WASM (clean build):
```bash
cd src && make clean && make
```

3. Copy the ONNX model to the build directory:
```bash
cp inference/model.onnx src/build/
```

4. Kill any existing server on port 8000 and start a new one:
```bash
pkill -f "python3 -m http.server 8000" 2>/dev/null || true
cd src/build && python3 -m http.server 8000 &
```

5. Report success and provide the URL:
```
Server running at http://localhost:8000
```
