"""Fetch safetensors header only (Range request), parse and output all keys. No full download."""
import json
import struct
import sys
import urllib.request

URL = "https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union/resolve/main/Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors"
OUT = "parse_fun_cn_header_result.txt"

def main():
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req2 = urllib.request.Request(URL, headers={"Range": "bytes=8-%d" % (8 + n - 1)})
    with urllib.request.urlopen(req2, timeout=60) as r:
        raw = r.read()
    meta = json.loads(raw.decode("utf-8"))
    keys = sorted([k for k in meta if k != "__metadata__"])
    lines = ["Total keys: %d" % len(keys), ""]
    for k in keys:
        v = meta[k]
        if isinstance(v, dict) and "shape" in v and "dtype" in v:
            lines.append("%s\t%s\t%s" % (k, v.get("dtype",""), v.get("shape","")))
        else:
            lines.append(k)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("OK written %s" % OUT)
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
