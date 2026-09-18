"""Debug script to trace the convergence loop."""
import numpy as np
from core.cnn_da_emd_olsb import (
    _get_cap_maps, compute_capacity, _get_bootstrap_positions,
    _build_recovery_side_info, BOOTSTRAP_N_PIXELS
)
from core.payload import prepare_payload, HEADER_SIZE_BYTES

# Same cover as diagnostic test
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

h, w = cover.shape[:2]
print(f"Cover: {cover.shape}")
print(f"Secret: {len(secret)} bytes")

# Step 1: Bootstrap
n_bootstrap = BOOTSTRAP_N_PIXELS
bootstrap_yx = _get_bootstrap_positions(h, w)
bootstrap_mask = np.zeros((h, w), dtype=bool)
bootstrap_mask[bootstrap_yx[:, 0], bootstrap_yx[:, 1]] = True
bootstrap_orig_lsbs = np.array([int(cover[y, x, 2]) & 1 for y, x in bootstrap_yx], dtype=np.uint8)

# Step 2: Capacity
upper = (cover & 0xF8).astype(np.uint8)
cls_r, cls_g, cls_b = _get_cap_maps(upper, 0.5, 0.5, 0.0, 0.33, 0.66, model=None)
cap_info = compute_capacity(cls_r, cls_g, cls_b, upper)

# Step 3: Get positions
all_emd = np.argwhere(cap_info['emd_mask'])
all_olsb = np.argwhere(cap_info['olsb_mask'])
emd_positions = all_emd[~bootstrap_mask[all_emd[:, 0], all_emd[:, 1]]] if len(all_emd) > 0 else np.empty((0, 2), dtype=np.intp)
olsb_positions = all_olsb[~bootstrap_mask[all_olsb[:, 0], all_olsb[:, 1]]] if len(all_olsb) > 0 else np.empty((0, 2), dtype=np.intp)

total_emd_bits_cap = len(emd_positions) * 2
usable_olsb_bits = len(olsb_positions) * 3
usable_body_capacity = total_emd_bits_cap + usable_olsb_bits

print(f"\nEMD positions (non-bootstrap): {len(emd_positions)}")
print(f"OLSB positions (non-bootstrap): {len(olsb_positions)}")
print(f"total_emd_bits_cap: {total_emd_bits_cap} bits = {total_emd_bits_cap//8} bytes")
print(f"usable_body_capacity: {usable_body_capacity} bits = {usable_body_capacity//8} bytes")

# Step 4: Prepare payload
payload_bytes_est = prepare_payload(secret, 'Pass123!', 0.33, 0.66, 0, gamma=0.0, location_map_data=None)
body_est = payload_bytes_est[HEADER_SIZE_BYTES:]
raw_body_bits = len(body_est) * 8

print(f"\nEncrypted payload (no side info): {len(body_est)} bytes = {raw_body_bits} bits")

# How many EMD positions needed?
emd_bits_needed = min(raw_body_bits, total_emd_bits_cap)
emd_bits_needed = (emd_bits_needed // 2) * 2
emd_used = emd_bits_needed // 2
olsb_bits_needed = raw_body_bits - emd_bits_needed
olsb_used = (olsb_bits_needed + 2) // 3

print(f"\nFor secret: emd_used={emd_used}, olsb_used={olsb_used}")

# Build side info for just those positions
side_info = _build_recovery_side_info(
    cover, emd_positions, emd_used, olsb_positions, olsb_used,
    bootstrap_yx, bootstrap_orig_lsbs
)
print(f"Side info size: {len(side_info)} bytes = {len(side_info)*8} bits")

# Side info uncompressed estimate
import zlib, struct
uncompressed_estimate = 13 + (emd_used * 6 + olsb_used * 3 + n_bootstrap + 7) // 8
print(f"Expected uncompressed: ~{uncompressed_estimate} bytes")

# Total body after including side info
payload_with_si = prepare_payload(secret, 'Pass123!', 0.33, 0.66, 0, gamma=0.0, location_map_data=side_info)
body_with_si = payload_with_si[HEADER_SIZE_BYTES:]
print(f"\nBody with side info: {len(body_with_si)} bytes = {len(body_with_si)*8} bits")
print(f"Carrier capacity:    {usable_body_capacity} bits = {usable_body_capacity//8} bytes")
print(f"Fits: {len(body_with_si)*8 <= usable_body_capacity}")
