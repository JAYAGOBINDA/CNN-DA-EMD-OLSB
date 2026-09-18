"""Debug cover recovery."""
import numpy as np
from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb

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

stego, stats = embed_cnn_da_emd_olsb(cover, secret, 'Pass123!', gamma=0.0)
print(f"Embed OK. Side info: {stats['recovery_side_info_bytes']} bytes")

extracted, recovered, meta = extract_cnn_da_emd_olsb(stego, 'Pass123!', gamma=0.0)
print(f"Extract OK. Secret match: {extracted == secret}")
print(f"Cover match: {np.array_equal(cover, recovered)}")

# Find differences
if not np.array_equal(cover, recovered):
    diff = cover.astype(int) - recovered.astype(int)
    nonzero = np.argwhere(diff != 0)
    print(f"\nNumber of different pixels: {len(np.unique(nonzero[:, :2], axis=0))}")
    print(f"Different pixel locations (first 10): {nonzero[:10]}")
    for coord in nonzero[:5]:
        y, x, c = coord
        print(f"  [{y},{x},{c}]: cover={cover[y,x,c]}, recovered={recovered[y,x,c]}, stego={stego[y,x,c]}")

# Check what side_info_data the extractor got
print(f"\nMetadata keys: {list(meta.keys())}")
side_info = meta.get('location_map_data')
print(f"location_map_data length: {len(side_info) if side_info else None}")
