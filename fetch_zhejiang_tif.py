#!/usr/bin/env python3
"""通过 HTTP Range 请求，从 Zenodo 上的大 zip 中只抽取浙江的 tif，
不下载整个压缩包。仅用标准库实现最小化的远程 zip 读取。"""
import os
import socket
import ssl
import struct
import sys
import time
import zlib
from urllib.request import Request, build_opener, HTTPSHandler, ProxyHandler

URL = "https://zenodo.org/records/18180184/files/CLCD_v01_2025_albert_province.zip?download=1"
HOST = "zenodo.org"
IP = "188.184.103.118"          # 直连 IP（本机本地 DNS 不可用）
TARGET_SUFFIX = "zhejiang.tif"  # 要抽取的成员
OUT = "/home/zsai/python_projects/prect500-idw/CLCD_v01_2025_albert_zhejiang.tif"

# 本地 DNS 解析不可用：把 zenodo.org 固定解析到已知 IP（SNI/Host 仍为域名，证书正常）
_orig_getaddrinfo = socket.getaddrinfo
def _patched(host, *args, **kwargs):
    if host == HOST:
        host = IP
    return _orig_getaddrinfo(host, *args, **kwargs)
socket.getaddrinfo = _patched

# 强制直连：忽略环境变量里已失效的本地代理（如 127.0.0.1:7897）
opener = build_opener(
    ProxyHandler({}),
    HTTPSHandler(context=ssl.create_default_context()),
)
opener.addheaders = []

def fetch_range(start, end, retries=8):
    """下载字节区间 [start, end]（含端点），带重试。"""
    last = None
    for attempt in range(retries):
        try:
            req = Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "curl/7.68"})
            with opener.open(req, timeout=120) as r:
                if r.status not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status}")
                buf = r.read()
            need = end - start + 1
            if len(buf) != need:
                raise RuntimeError(f"短读 {len(buf)}/{need}")
            return buf
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 15))
    raise last


def download_to_file(start, end, out_path, chunk=1 << 20):
    """把字节区间 [start, end] 分块（每块带重试）流式写入文件，支持断点续传。"""
    total = end - start + 1
    done = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    if done > total:
        done = 0
    mode = "r+b" if done else "wb"
    with open(out_path, mode) as f:
        f.seek(done)
        while done < total:
            s = start + done
            e = min(s + chunk - 1, end)
            f.write(fetch_range(s, e))
            done += e - s + 1
            print(f"\r  已下载 {done/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="", flush=True)
    print()

def get_total_size():
    req = Request(URL, method="HEAD", headers={"User-Agent": "curl/7.68"})
    with opener.open(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def main():
    total = get_total_size()
    print(f"zip 总大小: {total/1024/1024:.1f} MB")

    # 1) 读末尾找 EOCD（结束中央目录记录）
    tail_len = min(65557, total)
    tail = fetch_range(total - tail_len, total - 1)
    idx = tail.rfind(b"PK\x05\x06")
    if idx < 0:
        sys.exit("未找到 EOCD，可能是 ZIP64，需要扩展解析")
    eocd = tail[idx:idx + 22]
    (_, _, _, _, total_entries, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", eocd)
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        sys.exit("检测到 ZIP64 标记，需要扩展解析")
    print(f"成员数: {total_entries}, 中央目录: offset={cd_offset}, size={cd_size}")

    # 2) 下载中央目录并解析每个成员
    cd = fetch_range(cd_offset, cd_offset + cd_size - 1)
    p = 0
    members = []  # (name, method, comp_size, local_offset)
    while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
        (sig, ver_made, ver_need, flags, method, mtime, mdate, crc,
         comp_size, uncomp_size, name_len, extra_len, comment_len,
         disk_start, int_attr, ext_attr, local_off) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p+46])
        name = cd[p+46:p+46+name_len].decode("utf-8", "replace")
        members.append((name, method, comp_size, uncomp_size, local_off))
        p += 46 + name_len + extra_len + comment_len

    print("压缩包内成员:")
    for name, method, cs, us, off in members:
        print(f"  - {name}  (压缩 {cs/1024/1024:.1f}MB / 原始 {us/1024/1024:.1f}MB, method={method})")

    # 3) 定位目标成员
    target = next((m for m in members if m[0].lower().endswith(TARGET_SUFFIX)), None)
    if not target:
        sys.exit(f"未找到以 {TARGET_SUFFIX} 结尾的成员")
    name, method, comp_size, uncomp_size, local_off = target
    print(f"\n抽取: {name}  压缩 {comp_size/1024/1024:.1f}MB -> 原始 {uncomp_size/1024/1024:.1f}MB")

    # 4) 读本地文件头，确定实际数据起点（本地头的 name_len/extra_len 可能与中央目录不同）
    lh = fetch_range(local_off, local_off + 29)
    if lh[:4] != b"PK\x03\x04":
        sys.exit("本地文件头签名不符")
    ln_name = struct.unpack("<H", lh[26:28])[0]
    ln_extra = struct.unpack("<H", lh[28:30])[0]
    data_start = local_off + 30 + ln_name + ln_extra
    data_end = data_start + comp_size - 1

    # 5) 下载数据（分块+重试）并按需解压
    print("下载数据中...")
    if method == 0:
        # 未压缩存储：直接流式写入目标文件（支持断点续传）
        download_to_file(data_start, data_end, OUT)
    elif method == 8:
        tmp = OUT + ".deflate"
        download_to_file(data_start, data_end, tmp)
        with open(tmp, "rb") as f:
            data = zlib.decompress(f.read(), -15)
        with open(OUT, "wb") as f:
            f.write(data)
        os.remove(tmp)
    else:
        sys.exit(f"不支持的压缩方法 method={method}")

    size = os.path.getsize(OUT)
    print(f"完成 -> {OUT}  ({size/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
