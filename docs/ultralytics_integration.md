# Ultralytics / YOLO26 integration notes

> **Scope:** The repository provides a working standalone PyTorch attention block.
> The supplied source does **not** contain a modified Ultralytics framework, a complete detector YAML, pretrained weights, or a tested parser patch. The steps below are an adaptation guide, not a claim of drop-in compatibility.

## Tensor contract

```python
block = TCPELIA(c_x=128, c_z=256, c_r=64, iters=2)
y = block([x_cur, z_fb])
```

- `x_cur`: `[B, c_x, H, W]`, the feature to enhance.
- `z_fb`: `[B, c_z, Hf, Wf]`, the feedback feature. It is resized internally.
- `y`: `[B, c_x, H, W]`. `c_x` (not `c_r`) is the output channel count.
- `iters` is **5 in the original Python constructor**; pass `iters=2` explicitly if using the example configuration.

## Source YAML example

```yaml
- [[15, 10], 1, TCPELIA, [64, 2]]
```

This line comes from the provided source docstring. **It only works after the detector's model parser has been taught** (1) to recognize `TCPELIA`, (2) to receive two channel counts, and (3) to preserve the list of two input tensors at runtime.

Conceptually, the parser should read:

```python
# Illustrative parser adaptation; NOT a verbatim patch for all versions.
if m is TCPELIA:
    assert isinstance(f, (list, tuple)) and len(f) == 2
    c_x = ch[f[0]]
    c_z = ch[f[1]]
    c_r, iters = args
    args = [c_x, c_z, c_r, iters]
    c2 = c_x
```

The actual location and form of `parse_model`, the module registry, and graph indexing vary by Ultralytics version. Avoid automatically adding `TCPELIA` to a generic one-input Conv-style parser rule; doing so may mangle the two-input channel contract.

## Integration checklist

1. Place/import the module in your locally installed detector framework.
2. Register the `TCPELIA` symbol where the model YAML resolver can access it.
3. Update parser argument and output-channel inference specifically for its **two inputs**.
4. Ensure upstream feature tensor indexing provides both features with appropriate batch sizes.
5. Construct the model and run a forward/backward pass before training.
6. Check for graph export/inference compatibility separately; standalone PyTorch support does not imply ONNX/TensorRT support.

Never publish private datasets, local absolute paths, weights, or claimed benchmark scores without corresponding reproducible material and permission.
