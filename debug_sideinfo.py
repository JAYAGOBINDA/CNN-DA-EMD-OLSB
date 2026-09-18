"""Debug recovery side info parsing."""
import numpy as np
import zlib, struct
from core.cnn_da_emd_olsb import (
    embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb,
    _get_cap_maps, compute_capacity, _get_bootstrap_positions,
    _build_recovery_side_info, _apply_recovery_side_info, BOOTSTRAP_N_PIXELS
)
from core.payload import HEADER_SIZE_BYTES

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

# Embed
stego, stats = embed_cnn_da_emd_olsb(cover, secret, 'Pass123!', gamma=0.0)
print(f"Embed OK. Side info size: {stats['recovery_side_info_bytes']}")

# Simulate extraction to get the side info and positions
h, w = stego.shape[:2]
bootstrap_yx = _get_bootstrap_positions(h, w)
bootstrap_mask = np.zeros((h, w), dtype=bool)
bootstrap_mask[bootstrap_yx[:, 0], bootstrap_yx[:, 1]] = True

upper_stego = (stego & 0xF8).astype(np.uint8)
cls_r, cls_g, cls_b = _get_cap_maps(upper_stego, 0.5, 0.5, 0.0, 0.33, 0.66, model=None)
cap_info = compute_capacity(cls_r, cls_g, cls_b, upper_stego)

all_emd = np.argwhere(cap_info['emd_mask'])
all_olsb = np.argwhere(cap_info['olsb_mask'])
emd_positions = all_emd[~bootstrap_mask[all_emd[:, 0], all_emd[:, 1]]] if len(all_emd) > 0 else np.empty((0, 2))
olsb_positions = all_olsb[~bootstrap_mask[all_olsb[:, 0], all_olsb[:, 1]]] if len(all_olsb) > 0 else np.empty((0, 2))

# Also get positions from COVER (for comparison)
upper_cover = (cover & 0xF8).astype(np.uint8)
cls_r2, cls_g2, cls_b2 = _get_cap_maps(upper_cover, 0.5, 0.5, 0.0, 0.33, 0.66, model=None)
cap_info2 = compute_capacity(cls_r2, cls_g2, cls_b2, upper_cover)
all_emd2 = np.argwhere(cap_info2['emd_mask'])
emd_positions_cover = all_emd2[~bootstrap_mask[all_emd2[:, 0], all_emd2[:, 1]]] if len(all_emd2) > 0 else np.empty((0, 2))

print(f"\nEMD positions (from stego ): {len(emd_positions)}")
print(f"EMD positions (from cover): {len(emd_positions_cover)}")
print(f"Positions match: {np.array_equal(emd_positions, emd_positions_cover)}")

# Simulate extraction 
extracted, recovered, meta = extract_cnn_da_emd_olsb(stego, 'Pass123!', gamma=0.0)
side_info_bytes = meta.get('location_map_data')
print(f"\nSide info from extraction: {len(side_info_bytes)} bytes")

# Parse the side info
uncompressed = zlib.decompress(side_info_bytes)
version = uncompressed[0]
n_emd, n_olsb, n_boot = struct.unpack('!III', uncompressed[1:13])
print(f"Side info version: 0x{version:02x}")
print(f"n_emd_used={n_emd}, n_olsb_used={n_olsb}, n_bootstrap={n_boot}")

# For the specific failing pixel [5,75,1] (G channel):
y, x = 5, 75
cover_g = cover[y, x, 1]
stego_g = stego[y, x, 1]
recovered_g = recovered[y, x, 1]
print(f"\nPixel [{y},{x}] G channel:")
print(f"  cover={cover_g} (lower3={cover_g & 7})")
print(f"  stego={stego_g} (lower3={stego_g & 7})")
print(f"  recovered={recovered_g} (lower3={recovered_g & 7})")

# Check if [5,75] is in the emd_positions
idx = np.where((emd_positions[:, 0] == y) & (emd_positions[:, 1] == x))[0]
print(f"  Position [{y},{x}] is at emd_positions index: {idx}")

# Is the index within n_emd?
if len(idx) > 0:
    idx_val = idx[0]
    print(f"  index={idx_val}, n_emd_used={n_emd}: in range={idx_val < n_emd}")
