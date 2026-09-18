"""Test just Test 3 (large cover)."""
import numpy as np
from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb

cover_large = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(512):
    for j in range(512):
        cover_large[i, j] = [int(i / 2) % 256, int(j / 2) % 256, int((i + j) / 4) % 256]
cover_large = np.clip(cover_large, 0, 255).astype(np.uint8)

secret_large = b'A' * 500

try:
    stego3, stats3 = embed_cnn_da_emd_olsb(cover_large, secret_large, 'Pass123!', gamma=0.0)
    print('EMBED (large) OK!')
    print(f'  Usable cap: {stats3["usable_capacity_bytes"]} bytes')
    print(f'  Side info: {stats3["recovery_side_info_bytes"]} bytes')
    print(f'  converged: {stats3["recovery_side_info_bytes"] > 0}')

    extracted3, recovered3, meta3 = extract_cnn_da_emd_olsb(stego3, 'Pass123!', gamma=0.0)
    print(f'EXTRACT (large) OK! Got length: {len(extracted3)}')
    print(f'Secrets match: {extracted3 == secret_large}')
    print(f'Cover recovered exactly: {np.array_equal(cover_large, recovered3)}')
except Exception as e:
    import traceback
    print(f'ERROR: {e}')
    traceback.print_exc()
