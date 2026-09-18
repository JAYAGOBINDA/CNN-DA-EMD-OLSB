"""Diagnostic test for embed/extract functionality."""
import numpy as np
from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb

# Test 1: Small photographic-like image (smooth gradient)
cover = np.zeros((200, 300, 3), dtype=np.uint8)
for i in range(200):
    for j in range(300):
        cover[i, j] = [
            int(128 + 50 * np.sin(i / 20) + 20 * np.cos(j / 30)),
            int(100 + 40 * np.cos(i / 15) + 30 * np.sin(j / 25)),
            int(150 + 60 * np.sin((i + j) / 40))
        ]
cover = np.clip(cover, 0, 255).astype(np.uint8)

secret = b'Hello World! This is a test secret message for embedding.'
print(f'Cover: {cover.shape}, Secret: {len(secret)} bytes')

try:
    stego, stats = embed_cnn_da_emd_olsb(cover, secret, 'Pass123!', gamma=0.0)
    print('EMBED OK!')
    print(f'  Usable cap: {stats["usable_capacity_bytes"]} bytes')
    print(f'  Side info: {stats["recovery_side_info_bytes"]} bytes')
    print(f'  side_info converged: {stats["recovery_side_info_bytes"] > 0}')

    extracted, recovered, meta = extract_cnn_da_emd_olsb(stego, 'Pass123!', gamma=0.0)
    print(f'EXTRACT OK! Got: {extracted[:50]}')
    print(f'Secrets match: {extracted == secret}')
    print(f'Cover recovered exactly: {np.array_equal(cover, recovered)}')
except Exception as e:
    import traceback
    print(f'ERROR: {e}')
    traceback.print_exc()

print()

# Test 2: Random noisy image (hard case)
np.random.seed(42)
cover_noisy = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
secret2 = b'Test secret for noisy cover.'
print(f'Noisy cover: {cover_noisy.shape}, Secret: {len(secret2)} bytes')

try:
    stego2, stats2 = embed_cnn_da_emd_olsb(cover_noisy, secret2, 'Pass123!', gamma=0.0)
    print('EMBED (noisy) OK!')
    print(f'  Usable cap: {stats2["usable_capacity_bytes"]} bytes')
    print(f'  Side info: {stats2["recovery_side_info_bytes"]} bytes')

    extracted2, recovered2, meta2 = extract_cnn_da_emd_olsb(stego2, 'Pass123!', gamma=0.0)
    print(f'EXTRACT (noisy) OK! Got: {extracted2[:40]}')
    print(f'Secrets match: {extracted2 == secret2}')
    print(f'Cover recovered exactly: {np.array_equal(cover_noisy, recovered2)}')
except Exception as e:
    import traceback
    print(f'ERROR (noisy): {e}')
    traceback.print_exc()

print()

# Test 3: Large cover with large secret
print('Test 3: 512x512 cover, large secret...')
cover_large = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(512):
    for j in range(512):
        cover_large[i, j] = [int(i / 2) % 256, int(j / 2) % 256, int((i + j) / 4) % 256]
cover_large = np.clip(cover_large, 0, 255).astype(np.uint8)

secret_large = b'A' * 500  # 500 bytes
print(f'Large cover: {cover_large.shape}, Secret: {len(secret_large)} bytes')
try:
    stego3, stats3 = embed_cnn_da_emd_olsb(cover_large, secret_large, 'Pass123!', gamma=0.0)
    print('EMBED (large) OK!')
    print(f'  Usable cap: {stats3["usable_capacity_bytes"]} bytes')
    print(f'  Side info: {stats3["recovery_side_info_bytes"]} bytes')

    extracted3, recovered3, meta3 = extract_cnn_da_emd_olsb(stego3, 'Pass123!', gamma=0.0)
    print(f'EXTRACT (large) OK! Got length: {len(extracted3)}')
    print(f'Secrets match: {extracted3 == secret_large}')
    print(f'Cover recovered exactly: {np.array_equal(cover_large, recovered3)}')
except Exception as e:
    import traceback
    print(f'ERROR (large): {e}')
    traceback.print_exc()
