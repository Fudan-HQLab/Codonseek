import hashlib
import sqlite3
import os

def sqlite_deduplicate(input_path, output_path, batch_size=10000):
    """使用SQLite数据库进行磁盘级去重"""
    DB_PATH = "temp_seq.db"
    
    # 初始化数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sequences (
            seq_md5 BLOB PRIMARY KEY,
            full_text TEXT
        )
    """)
    
    # 批量插入数据
    buffer = []
    with open(input_path, 'r') as fin:
        for line in fin:
            try:
                seq = line.split("| Seq: ")[1].strip()
            except IndexError:
                continue
            
            # 生成MD5哈希
            seq_hash = hashlib.md5(seq.encode()).digest()
            buffer.append((seq_hash, line))
            
            # 批量提交
            if len(buffer) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO sequences VALUES (?, ?)",
                    buffer
                )
                conn.commit()
                buffer = []
        
        # 提交剩余数据
        if buffer:
            cursor.executemany(
                "INSERT OR IGNORE INTO sequences VALUES (?, ?)",
                buffer
            )
            conn.commit()
    
    # 输出结果
    cursor.execute("SELECT full_text FROM sequences ORDER BY rowid")
    with open(output_path, 'w') as fout:
        for row in cursor:
            fout.write(row[0])
    
    # 清理
    conn.close()
    os.remove(DB_PATH)

# 根据内存情况选择执行方法
if __name__ == "__main__":
    # for num in range(1,11):
    #     print(num)
        input_file = "combined_texts.txt"
        output_file = "undealtotal_top_heap.txt"
    
    # 内存充足时（>32GB）
    # deduplicate_large_file(input_file, output_file)
    
    # 内存有限时
        sqlite_deduplicate(input_file, output_file, batch_size=50000)