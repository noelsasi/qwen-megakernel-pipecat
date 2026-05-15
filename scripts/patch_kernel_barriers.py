"""
Patch torch_bindings.cpp to expose a reset_barriers() op.
This lets Python reset d_barrier_counter/sense/kv_flag/attn_flag
before each decode() call, eliminating the barrier race in consecutive calls.

Run from repo root on the GPU server:
    python scripts/patch_kernel_barriers.py
    cd qwen_megakernel && python build.py && cd ..
"""
import re, sys, os

BINDINGS = "qwen_megakernel/csrc/torch_bindings.cpp"
KERNEL   = "qwen_megakernel/csrc/kernel.cu"

# ── 1. Add reset function to kernel.cu (after ensure_barrier_alloc) ──────────
with open(KERNEL) as f:
    ksrc = f.read()

RESET_DECL = '''
extern "C" void reset_barriers() {
  ensure_barrier_alloc();
  cudaMemset(d_barrier_counter, 0, sizeof(unsigned int));
  cudaMemset(d_barrier_sense,   0, sizeof(unsigned int));
  cudaMemset(d_kv_flag,         0, sizeof(unsigned int));
  cudaMemset(d_attn_flag,       0, sizeof(unsigned int));
  cudaDeviceSynchronize();
}
'''

if 'void reset_barriers()' in ksrc:
    print("kernel.cu: reset_barriers already present")
else:
    # Insert right before the first extern "C" void launch_ldg_decode_direct
    target = 'extern "C" void launch_ldg_decode_direct('
    if target not in ksrc:
        print(f"ERROR: could not find '{target}' in kernel.cu", file=sys.stderr)
        sys.exit(1)
    ksrc = ksrc.replace(target, RESET_DECL + '\n' + target, 1)
    with open(KERNEL, 'w') as f:
        f.write(ksrc)
    print("kernel.cu: reset_barriers() added")

# ── 2. Add binding to torch_bindings.cpp ─────────────────────────────────────
with open(BINDINGS) as f:
    bsrc = f.read()

# Add extern declaration after the last existing extern "C" declaration
EXTERN_DECL = '\nextern "C" void reset_barriers();\n'
RESET_FN = '''
void reset_barriers_op() {
  reset_barriers();
}
'''
OP_REG = '''  ops.def("reset_barriers() -> ()");
  ops.impl("reset_barriers", torch::kCUDA, &reset_barriers_op);
'''

if 'reset_barriers_op' in bsrc:
    print("torch_bindings.cpp: reset_barriers_op already present")
else:
    # Add extern decl before TORCH_LIBRARY_EXPAND
    anchor = 'TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME'
    if anchor not in bsrc:
        print(f"ERROR: could not find '{anchor}' in torch_bindings.cpp", file=sys.stderr)
        sys.exit(1)
    bsrc = bsrc.replace(anchor, EXTERN_DECL + RESET_FN + '\n' + anchor, 1)
    # Register op — insert after ops.impl("decode", ...)
    reg_anchor = 'ops.impl("decode", torch::kCUDA, &decode);'
    if reg_anchor not in bsrc:
        print(f"ERROR: could not find decode impl line", file=sys.stderr)
        sys.exit(1)
    bsrc = bsrc.replace(reg_anchor, reg_anchor + '\n' + OP_REG, 1)
    with open(BINDINGS, 'w') as f:
        f.write(bsrc)
    print("torch_bindings.cpp: reset_barriers_op registered")

print("Done. Now run: cd qwen_megakernel && python build.py && cd ..")
