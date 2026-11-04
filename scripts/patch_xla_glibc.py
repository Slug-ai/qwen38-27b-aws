#!/usr/bin/env python3
"""Let torch-xla's _XLAC load on Amazon Linux 2023, whose glibc is 2.34.

The 2.9.0 wheel is tagged manylinux_2_28 but binds hypot and hypotf to GLIBC_2.35, so importing it fails,
even in the Neuron AMI's own venvs. This drops that one version requirement, and those two symbols then
bind to the host's hypot, which behaves the same apart from errno handling.
"""

import argparse
import os
import struct
from pathlib import Path

SHT_DYNSYM = 11
SHT_GNU_VERNEED = 0x6ffffffe
SHT_GNU_VERSYM = 0x6fffffff
VER_FLG_WEAK = 0x2
VER_NDX_GLOBAL = 1
HIDDEN_BIT = 0x8000


def read_cstr(data: bytearray, offset: int) -> str:
    return data[offset : data.index(0, offset)].decode()


def sections(data: bytearray) -> list[dict]:
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    out = []
    for i in range(shnum):
        _, kind, _, _, offset, size, link, info, _, _ = struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        out.append({"type": kind, "offset": offset, "size": size, "link": link, "info": info})
    return out


def patch(data: bytearray, version: str) -> list[str]:
    all_sections = sections(data)
    by_type = {s["type"]: s for s in all_sections}
    verneed, versym, dynsym = by_type[SHT_GNU_VERNEED], by_type[SHT_GNU_VERSYM], by_type[SHT_DYNSYM]
    verneed_strings = all_sections[verneed["link"]]["offset"]
    dynsym_strings = all_sections[dynsym["link"]]["offset"]

    index = None
    need = verneed["offset"]
    for _ in range(verneed["info"]):
        _, aux_count, _, aux_offset, next_need = struct.unpack_from("<HHIII", data, need)
        aux = need + aux_offset
        for _ in range(aux_count):
            _, flags, other, name, next_aux = struct.unpack_from("<IHHII", data, aux)
            if read_cstr(data, verneed_strings + name) == version:
                index = other
                struct.pack_into("<H", data, aux + 4, flags | VER_FLG_WEAK)
            aux += next_aux
        need += next_need
    if index is None:
        return []

    patched = []
    for i in range(dynsym["size"] // 24):
        entry = versym["offset"] + i * 2
        if struct.unpack_from("<H", data, entry)[0] & ~HIDDEN_BIT == index:
            struct.pack_into("<H", data, entry, VER_NDX_GLOBAL)
            name = struct.unpack_from("<I", data, dynsym["offset"] + i * 24)[0]
            patched.append(read_cstr(data, dynsym_strings + name))
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, help="Path to _XLAC*.so inside the venv.")
    parser.add_argument("--version", default="GLIBC_2.35")
    args = parser.parse_args()

    data = bytearray(args.library.read_bytes())
    patched = patch(data, args.version)
    print(f"cleared {args.version} from: {patched or 'nothing, already patched'}")
    # Write a new file, because uv hard-links venv files to its shared cache.
    temp = args.library.with_suffix(".patched")
    temp.write_bytes(data)
    temp.chmod(args.library.stat().st_mode)
    os.replace(temp, args.library)


if __name__ == "__main__":
    main()
