import os
import numpy as np
import h5py
from datetime import datetime

DB_PATH = "db_embedding/embed_s.h5"

def print_separator():
    print("=" * 80)

def print_header(title):
    print_separator()
    print(f" {title.upper()} ".center(80, "="))
    print_separator()

def analyze_embedding_quality(embedding):
    """Phân tích chất lượng embedding"""
    try:
        # Chuyển đổi sang numpy array và đảm bảo kiểu dữ liệu
        embedding = np.asarray(embedding, dtype=np.float32)
        
        # Kiểm tra nếu embedding rỗng
        if embedding.size == 0:
            return {'quality': 'Empty', 'dims': (0,), 'error': 'Empty embedding'}

        norm = float(np.linalg.norm(embedding))
        mean_val = float(np.mean(embedding))
        std_val = float(np.std(embedding))
        min_val = float(np.min(embedding))
        max_val = float(np.max(embedding))
        
        # Đánh giá chất lượng
        quality = "Good"
        error_msg = None
        
        # Kiểm tra NaN và inf
        has_nan = np.any(np.isnan(embedding))
        has_inf = np.any(np.isinf(embedding))
        if has_nan or has_inf:
            quality = "Corrupted"
            error_msg = "Contains NaN or Inf"
        elif norm < 0.9 or norm > 1.1:
            quality = "Suspicious"
            error_msg = f"Unusual norm: {norm:.3f}"
        
        result = {
            'dims': embedding.shape,
            'norm': norm,
            'mean': mean_val,
            'std': std_val,
            'range': (min_val, max_val),
            'quality': quality
        }
        
        if error_msg:
            result['error'] = error_msg
            
        return result
        
    except Exception as e:
        return {
            'quality': 'Error',
            'dims': 'Unknown',
            'error': str(e)
        }

def check_h5_database():
    """Kiểm tra và phân tích database H5"""
    print_header("HDF5 FACE RECOGNITION DATABASE CHECKER")
    
    if not os.path.exists(DB_PATH):
        print("❌ Database không tồn tại.")
        return
    
    try:
        with h5py.File(DB_PATH, 'r') as f:
            # Đọc metadata
            if 'metadata' in f:
                print(f"📁 File: {DB_PATH}")
                print(f"📝 Phiên bản: {f['metadata'].attrs.get('version', 'N/A')}")
                print(f"⏰ Tạo lúc: {f['metadata'].attrs.get('created_at', 'N/A')}")
                print(f"⏰ Cập nhật lần cuối: {f['metadata'].attrs.get('updated_at', 'N/A')}")
                print(f"📄 Mô tả: {f['metadata'].attrs.get('description', 'N/A')}")
            else:
                print(f"📁 File: {DB_PATH}")
                print("⚠️ Không có metadata")
            
            # Đọc danh sách tên
            if 'names' in f:
                raw_names = list(f['names'][()])
                names = []
                for n in raw_names:
                    if isinstance(n, bytes):
                        try:
                            names.append(n.decode('utf-8'))
                        except Exception:
                            names.append(n.decode('latin1', errors='ignore'))
                    else:
                        names.append(str(n))
                print(f"👥 Số người: {len(names)}")
            else:
                print("❌ Không có danh sách tên")
                return
            
            print(f"💾 Kích thước file: {os.path.getsize(DB_PATH) / 1024:.1f} KB")
            
            # Kiểm tra cấu trúc
            has_embeddings = 'embeddings' in f
            has_avg_embeddings = 'avg_embeddings' in f
            
            if has_embeddings:
                print("✅ Có embeddings")
            else:
                print("❌ Không có embeddings")
                
            if has_avg_embeddings:
                print("✅ Có average embeddings")
            else:
                print("❌ Không có average embeddings")
            
            print("\n👥 CHI TIẾT TỪNG NGƯỜI:")
            print("-" * 80)
            
            total_samples = 0
            
            for i, name in enumerate(names):
                print(f"\n🔹 Người thứ {i+1}: {name}")
                
                # Kiểm tra embeddings
                emb_path = f"embeddings/{name}"
                if emb_path in f:
                    embeddings = f[emb_path][()]
                    num_samples = embeddings.shape[0] if embeddings.ndim > 1 else 1
                    total_samples += num_samples
                    
                    print(f"   📸 Số lượng mẫu: {num_samples}")
                    
                    if 'updated_at' in f[emb_path].attrs:
                        print(f"   ⏰ Cập nhật: {f[emb_path].attrs['updated_at']}")
                    
                    # Phân tích mẫu
                    if num_samples > 0:
                        sample_count = min(3, num_samples)
                        for j in range(sample_count):
                            if embeddings.ndim > 1:
                                emb = embeddings[j]
                            else:
                                emb = embeddings
                                
                            quality_info = analyze_embedding_quality(emb)
                            error_str = f" | Error: {quality_info['error']}" if 'error' in quality_info else ""
                            print(f"   🔸 Mẫu {j+1}: {quality_info['dims']} | Norm: {quality_info['norm']:.3f} | Quality: {quality_info['quality']}{error_str}")
                        
                        if num_samples > 3:
                            print(f"   ... và {num_samples-3} mẫu khác")
                else:
                    print("   ❌ Không có embeddings")
                
                # Kiểm tra average embedding
                avg_path = f"avg_embeddings/{name}"
                if avg_path in f:
                    avg_emb = f[avg_path][()]
                    quality_info = analyze_embedding_quality(avg_emb)
                    error_str = f" | Error: {quality_info['error']}" if 'error' in quality_info else ""
                    print(f"   🎯 Average: {quality_info['dims']} | Norm: {quality_info['norm']:.3f} | Quality: {quality_info['quality']}{error_str}")
                    
                    # Lấy thông tin chất lượng từ thuộc tính
                    if 'quality' in f[avg_path].attrs:
                        quality = f[avg_path].attrs['quality']
                        norm = f[avg_path].attrs.get('norm', 0.0)
                        print(f"   ⚠️  Đánh giá: {quality.upper()} | Norm: {norm:.3f}")
                else:
                    print("   ❌ Không có average embedding")
            
            # Tổng kết
            print("\n" + "=" * 80)
            print(f"📊 TỔNG KẾT:")
            print(f"   👥 Số người: {len(names)}")
            print(f"   🔢 Tổng số mẫu: {total_samples}")
            print(f"   💾 Kích thước file: {os.path.getsize(DB_PATH) / 1024:.1f} KB")
        
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc database: {e}")

if __name__ == "__main__":
    check_h5_database()