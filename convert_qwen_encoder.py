"""
Qwen3 4B Text Encoder 変換スクリプト

このスクリプトは、abliterated版のQwen3テキストエンコーダを
ForgeNeoで使用可能な形式に変換します。

両ファイルは同じテンソルキー(398個)、形状、データ型(bfloat16)を持っていますが、
メタデータが異なっています:
- ForgeNeo互換版: メタデータなし
- abliterated版: {"format": "pt"}

このスクリプトはメタデータを削除して新しいファイルを作成します。
"""

import json
import safetensors.torch as st
import torch

def convert_qwen_encoder(input_path: str, output_path: str):
    """
    Qwen3 text encoderをForgeNeo互換形式に変換します。
    
    Args:
        input_path: 変換元のsafetensorsファイルパス
        output_path: 変換後のsafetensorsファイル出力パス
    """
    print(f"Loading: {input_path}")
    
    # safetensorsを読み込み
    tensors = {}
    with st.safe_open(input_path, framework='pt') as f:
        metadata = f.metadata()
        print(f"Original metadata: {json.dumps(metadata, indent=2) if metadata else None}")
        
        keys = list(f.keys())
        print(f"Number of tensors: {len(keys)}")
        
        for key in keys:
            tensors[key] = f.get_tensor(key)
    
    # メタデータなしで保存
    print(f"\nSaving to: {output_path}")
    print("Saving without metadata...")
    st.save_file(tensors, output_path)
    
    # 検証
    print("\n=== Verification ===")
    with st.safe_open(output_path, framework='pt') as f:
        new_metadata = f.metadata()
        new_keys = list(f.keys())
        print(f"New metadata: {json.dumps(new_metadata, indent=2) if new_metadata else None}")
        print(f"Number of tensors: {len(new_keys)}")
        
        # キーが同じか確認
        if set(keys) == set(new_keys):
            print("✓ All tensor keys preserved")
        else:
            print("✗ Tensor keys mismatch!")
            return False
        
        # いくつかのテンソルをサンプルチェック
        sample_keys = ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight', 'model.norm.weight']
        for key in sample_keys:
            original = tensors[key]
            new = f.get_tensor(key)
            if original.shape == new.shape and original.dtype == new.dtype:
                print(f"✓ {key}: shape={new.shape}, dtype={new.dtype}")
            else:
                print(f"✗ {key}: mismatch!")
                return False
    
    print("\n✓ Conversion completed successfully!")
    return True


if __name__ == "__main__":
    import sys
    
    # デフォルトパス
    input_file = r'D:\USERFILES\GitHub\sd-webui-forge-classic-neo\models\text_encoder\qwen3_4b_abliterated_fp16.safetensors'
    output_file = r'D:\USERFILES\GitHub\sd-webui-forge-classic-neo\models\text_encoder\qwen3_4b_abliterated_fp16_converted.safetensors'
    
    # コマンドライン引数があれば使用
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    
    convert_qwen_encoder(input_file, output_file)
