import os
import numpy as np
from datetime import datetime

DB_PATH = "embeddings_db.npz"
ADVANCED_DB_PATH = "embeddings_advanced_db.npz"

def print_separator():
    print("=" * 80)

def print_header(title):
    print_separator()
    print(f" {title.upper()} ".center(80, "="))
    print_separator()

def analyze_embedding_quality(embedding):
    """Phân tích chất lượng embedding"""
    try:
        if isinstance(embedding, dict):
            # Multiple model embeddings
            total_size = sum(len(v) for v in embedding.values())
            models = list(embedding.keys())
            return f"Multi-model ({', '.join(models)}) | Total dims: {total_size}"
        else:
            # Chuyển đổi sang numpy array và đảm bảo kiểu dữ liệu
            embedding = np.asarray(embedding, dtype=np.float32)
            
            # Kiểm tra nếu embedding rỗng
            if embedding.size == 0:
                return {'quality': 'Empty', 'dims': (0,), 'error': 'Empty embedding'}
            
            # Tính toán các thông số
            norm = float(np.linalg.norm(embedding))
            mean_val = float(np.mean(embedding))
            std_val = float(np.std(embedding))
            min_val = float(np.min(embedding))
            max_val = float(np.max(embedding))
            
            # Đánh giá chất lượng
            quality = "Good"
            error_msg = None
            
            # Kiểm tra NaN và inf một cách an toàn
            try:
                has_nan = np.any(np.isnan(embedding))
                has_inf = np.any(np.isinf(embedding))
                if has_nan or has_inf:
                    quality = "Corrupted"
                    error_msg = "Contains NaN or Inf"
            except (TypeError, ValueError):
                quality = "Unknown"
                error_msg = "Cannot check NaN/Inf"
            
            if norm < 0.5 or norm > 2.0:
                if quality != "Corrupted":
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

def check_single_db(db_path, db_name):
    """Kiểm tra một database cụ thể"""
    if not os.path.exists(db_path):
        print(f"❌ Database không tồn tại: {db_path}")
        return False
    
    try:
        data = np.load(db_path, allow_pickle=True)
        names = list(data["names"])
        embeddings = data["embeddings"]
        
        print_header(f"DATABASE: {db_name}")
        print(f"📁 File: {db_path}")
        print(f"📊 Tổng số người: {len(names)}")
        print(f"💾 File size: {os.path.getsize(db_path) / 1024:.1f} KB")
        print(f"⏰ Cập nhật lần cuối: {datetime.fromtimestamp(os.path.getmtime(db_path)).strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Kiểm tra avg_embeddings nếu có
        has_avg_embs = "avg_embeddings" in data.keys()
        if has_avg_embs:
            avg_embeddings = data["avg_embeddings"]
            print(f"✅ Có average embeddings")
        else:
            print(f"⚠️  Không có average embeddings")
        
        print()
        print("👥 CHI TIẾT TỪNG NGƯỜI:")
        print("-" * 80)
        
        for i, name in enumerate(names):
            print(f"\n🔹 Người thứ {i+1}: {name}")
            
            # Phân tích embeddings
            person_embs = embeddings[i]
            if isinstance(person_embs, (list, np.ndarray)) and len(person_embs) > 0:
                try:
                    if isinstance(person_embs[0], np.ndarray) or (hasattr(person_embs, 'shape') and len(person_embs.shape) > 1):
                        # Multiple embeddings
                        if hasattr(person_embs, '__len__'):
                            print(f"   📸 Số lượng mẫu: {len(person_embs)}")
                        else:
                            print(f"   📸 Embedding array shape: {person_embs.shape}")
                        
                        # Phân tích từng embedding
                        sample_count = min(3, len(person_embs)) if hasattr(person_embs, '__len__') else 1
                        for j in range(sample_count):
                            try:
                                emb = person_embs[j] if hasattr(person_embs, '__getitem__') else person_embs
                                quality_info = analyze_embedding_quality(emb)
                                if isinstance(quality_info, dict):
                                    error_str = f" | Error: {quality_info['error']}" if 'error' in quality_info else ""
                                    print(f"   🔸 Mẫu {j+1}: {quality_info['dims']} | Norm: {quality_info.get('norm', 'N/A')} | Quality: {quality_info['quality']}{error_str}")
                                else:
                                    print(f"   🔸 Mẫu {j+1}: {quality_info}")
                            except Exception as e:
                                print(f"   🔸 Mẫu {j+1}: Lỗi - {str(e)}")
                        
                        if hasattr(person_embs, '__len__') and len(person_embs) > 3:
                            print(f"   ... và {len(person_embs)-3} mẫu khác")
                            
                        # Thống kê tổng quan
                        try:
                            all_norms = []
                            for emb in person_embs:
                                try:
                                    emb_arr = np.asarray(emb, dtype=np.float32)
                                    norm = float(np.linalg.norm(emb_arr))
                                    all_norms.append(norm)
                                except:
                                    continue
                            
                            if all_norms:
                                print(f"   📈 Norm range: {min(all_norms):.3f} - {max(all_norms):.3f} (avg: {np.mean(all_norms):.3f})")
                            else:
                                print(f"   📈 Không thể tính norm statistics")
                        except Exception as e:
                            print(f"   📈 Lỗi tính norm: {str(e)}")
                            
                    else:
                        # Single embedding hoặc format khác
                        quality_info = analyze_embedding_quality(person_embs)
                        if isinstance(quality_info, dict):
                            error_str = f" | Error: {quality_info['error']}" if 'error' in quality_info else ""
                            print(f"   📸 Single embedding: {quality_info['dims']} | Norm: {quality_info.get('norm', 'N/A')} | Quality: {quality_info['quality']}{error_str}")
                        else:
                            print(f"   📸 Single embedding: {quality_info}")
                            
                except Exception as e:
                    print(f"   ❌ Lỗi phân tích embedding: {str(e)}")
                    print(f"   🔍 Type: {type(person_embs)} | Length: {len(person_embs) if hasattr(person_embs, '__len__') else 'N/A'}")
            else:
                print(f"   ❌ Embedding không hợp lệ: {type(person_embs)}")
            
            # Kiểm tra average embedding nếu có
            if has_avg_embs and i < len(avg_embeddings):
                try:
                    avg_emb = avg_embeddings[i]
                    avg_quality = analyze_embedding_quality(avg_emb)
                    if isinstance(avg_quality, dict):
                        error_str = f" | Error: {avg_quality['error']}" if 'error' in avg_quality else ""
                        print(f"   🎯 Average: {avg_quality['dims']} | Norm: {avg_quality.get('norm', 'N/A')} | Quality: {avg_quality['quality']}{error_str}")
                    else:
                        print(f"   🎯 Average: {avg_quality}")
                except Exception as e:
                    print(f"   🎯 Average: Lỗi - {str(e)}")
        
        print()
        print_separator()
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc database: {str(e)}")
        return False

def check_all_databases():
    """Kiểm tra tất cả databases có sẵn"""
    print_header("FACE RECOGNITION DATABASE CHECKER")
    print(f"🕐 Thời gian kiểm tra: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Tìm tất cả các file database
    databases = []
    if os.path.exists(DB_PATH):
        databases.append((DB_PATH, "Standard Database"))
    if os.path.exists(ADVANCED_DB_PATH):
        databases.append((ADVANCED_DB_PATH, "Advanced Database"))
    
    # Tìm các database khác
    for file in os.listdir("."):
        if file.endswith("_db.npz") and file not in [DB_PATH, ADVANCED_DB_PATH]:
            databases.append((file, f"Custom Database ({file})"))
    
    if not databases:
        print("❌ Không tìm thấy database nào!")
        print("💡 Hãy chạy data_capture.py để tạo database trước.")
        return
    
    print(f"🔍 Tìm thấy {len(databases)} database(s):")
    for db_path, db_name in databases:
        print(f"  - {db_name}: {db_path}")
    print()
    
    # Kiểm tra từng database
    for db_path, db_name in databases:
        check_single_db(db_path, db_name)
        print("\n")
    
    # Tổng kết
    print_header("TỔNG KẾT")
    total_people = 0
    for db_path, db_name in databases:
        if os.path.exists(db_path):
            try:
                data = np.load(db_path, allow_pickle=True)
                people_count = len(data["names"])
                total_people += people_count
                print(f"📊 {db_name}: {people_count} người")
            except:
                print(f"❌ {db_name}: Lỗi đọc file")
    
    print(f"\n🎯 Tổng cộng: {total_people} người được ghi danh")
    
    # Gợi ý
    print("\n💡 GỢI Ý:")
    print("  - Nếu quality = 'Suspicious', hãy ghi danh lại")
    print("  - Nếu quality = 'Corrupted', hãy xóa và ghi danh lại")
    print("  - Norm tốt nên trong khoảng 0.8 - 1.2")
    print("  - Nhiều mẫu (>15) sẽ cho độ chính xác cao hơn")

if __name__ == "__main__":
    check_all_databases()
