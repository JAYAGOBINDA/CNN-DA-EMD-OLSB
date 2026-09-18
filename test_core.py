"""Quick diagnostic test for embedding and extraction."""
import numpy as np
import traceback

print("Loading model...")
from models.cnn_da_emd_olsb_model import CNNDAEMDOLSBModel
from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb, _get_cap_maps, compute_capacity
from utils.image_utils import optimize_secret_image

model = CNNDAEMDOLSBModel()
print(f"Model loaded. CNN trained: {model._cnn_trained}")

# Test 1: Small cover with text
print("\n=== Test 1: Small Cover (100x100) with text ===")
cover = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
secret = b'Hello World Test'
try:
    stego, stats = model.embed(cover, secret, password='test123')
    print(f"Embed SUCCESS: stego shape={stego.shape}")
    extracted, recovered, meta = model.extract(stego, password='test123')
    print(f"Extract SUCCESS: text={extracted.decode('utf-8', errors='replace')}")
    print(f"Cover recovered exactly: {np.array_equal(cover, recovered)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# Test 2: Medium cover with text
print("\n=== Test 2: Medium Cover (256x256) with text ===")
cover2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
try:
    stego2, stats2 = model.embed(cover2, secret, password='test123')
    print(f"Embed SUCCESS: stego shape={stego2.shape}")
    extracted2, recovered2, meta2 = model.extract(stego2, password='test123')
    print(f"Extract SUCCESS: text={extracted2.decode('utf-8', errors='replace')}")
    print(f"Cover recovered exactly: {np.array_equal(cover2, recovered2)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# Test 3: Image payload - small secret into large cover
print("\n=== Test 3: Image payload - 64x64 secret into 512x512 cover ===")
cover3 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
secret_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
try:
    upper_c = (cover3 & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper_c, 0.5, 0.5, 0.6, 0.33, 0.66, model=model._cnn_model)
    cap_info = compute_capacity(cls_r, cls_g, cls_b, upper_c)
    usable_bytes = cap_info['usable_capacity_bytes']
    max_cap = max(64, int((usable_bytes - 64) * 0.12))
    print(f"Cover 512x512 capacity: usable={usable_bytes} bytes, max_payload={max_cap} bytes")
    
    opt_bytes, opt_w, opt_h = optimize_secret_image(secret_img, max_cap)
    print(f"Optimized secret: {opt_w}x{opt_h}, size={len(opt_bytes)} bytes")
    
    stego3, stats3 = model.embed(cover3, opt_bytes, password='test123', payload_type=1)
    print(f"Embed SUCCESS: stego shape={stego3.shape}")
    
    extracted3, recovered3, meta3 = model.extract(stego3, password='test123')
    print(f"Extract SUCCESS: extracted {len(extracted3)} bytes")
    print(f"Cover recovered exactly: {np.array_equal(cover3, recovered3)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# Test 4: Large secret image into small cover
print("\n=== Test 4: LARGE secret (1024x768) into small cover (256x256) ===")
cover4 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
secret_img4 = np.random.randint(0, 256, (768, 1024, 3), dtype=np.uint8)
try:
    upper_c4 = (cover4 & 0xF8).astype(np.uint8)
    cls_r4, cls_g4, cls_b4 = _get_cap_maps(upper_c4, 0.5, 0.5, 0.6, 0.33, 0.66, model=model._cnn_model)
    cap_info4 = compute_capacity(cls_r4, cls_g4, cls_b4, upper_c4)
    usable_bytes4 = cap_info4['usable_capacity_bytes']
    max_cap4 = max(64, int((usable_bytes4 - 64) * 0.12))
    print(f"Cover 256x256 capacity: usable={usable_bytes4} bytes, max_payload={max_cap4} bytes")
    
    opt_bytes4, opt_w4, opt_h4 = optimize_secret_image(secret_img4, max_cap4)
    print(f"Optimized secret: {opt_w4}x{opt_h4}, size={len(opt_bytes4)} bytes")
    
    stego4, stats4 = model.embed(cover4, opt_bytes4, password='test123', payload_type=1)
    print(f"Embed SUCCESS")
    
    extracted4, recovered4, meta4 = model.extract(stego4, password='test123')
    print(f"Extract SUCCESS: extracted {len(extracted4)} bytes")
    print(f"Cover recovered exactly: {np.array_equal(cover4, recovered4)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("\n=== ALL TESTS COMPLETE ===")
