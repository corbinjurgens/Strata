"""Offline packer coverage for UD-Q2_K_XL's mixed tensor formats, without model downloads or a GPU."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_iq_pack import GGUFWriter, Q, iq_pack, quants, run_pack, stored
from gguf import GGML_QUANT_SIZES


def native_bytes(kind, shape, seed):
    """Small, nonzero native blocks; gguf-py has no encoder for these K/IQ formats."""
    block_size, block_bytes = GGML_QUANT_SIZES[kind]
    assert shape[-1] % block_size == 0
    blocks = np.random.default_rng(seed).integers(
        0, 256, (int(np.prod(shape)) // block_size, block_bytes), dtype=np.uint8)
    scale = np.array([0.03125], dtype="<f2").view(np.uint8)
    # Q6_K stores its half scale last; the other fixture formats store it first.
    if kind == Q.Q6_K:
        blocks[:, -2:] = scale
    else:
        blocks[:, :2] = scale
    if kind in (Q.Q4_K, Q.Q5_K):
        blocks[:, 2:4] = scale                         # finite dmin as well as d
    return blocks.reshape(*shape[:-1], shape[-1] // block_size * block_bytes)


def q2_shards(root):
    """Two miniature layers preserve the real pack's format roles, split over three shards.

    Gate/up use IQ2_XS in most real layers and IQ3_XXS in one; down uses IQ4_NL.
    The split inside layer 0 also exercises independent gate/up/down shard offsets.
    These are packing fixtures, not numerically representative model weights.
    """
    tensors = {}

    def native(name, kind, shape=(4, 256)):
        tensors[name] = (native_bytes(kind, shape, len(tensors) + 1), kind)
        return name

    def quantized(name, kind, shape=(4, 256)):
        values = np.linspace(-0.3, 0.7, int(np.prod(shape)), dtype=np.float32).reshape(shape)
        tensors[name] = (quants.quantize(values, kind), kind)
        return name

    shards = [
        [native("output.weight", Q.Q4_K), native("token_embd.weight", Q.Q5_K),
         native("blk.0.attn_qkv.weight", Q.Q5_K), native("blk.0.attn_output.weight", Q.Q6_K),
         quantized("blk.0.ffn_gate_shexp.weight", Q.Q8_0),
         quantized("blk.0.hc_attn_down.weight", Q.Q8_0)],
        [native("per_layer_token_embd.weight", Q.IQ4_NL, (64, 32)),
         quantized("blk.1.ple_key.weight", Q.Q8_0), quantized("blk.1.ple_value.weight", Q.Q8_0)],
        [],
    ]
    for layer, kind in enumerate((Q.IQ2_XS, Q.IQ3_XXS)):
        shards[layer].append(quantized(f"blk.{layer}.ffn_gate_inp.weight", Q.F32))
        for role in ("gate", "up", "down"):
            shape = (4, 256, 32) if role == "down" else (4, 32, 256)
            name = native(f"blk.{layer}.ffn_{role}_exps.weight", Q.IQ4_NL if role == "down" else kind, shape)
            shards[2 if layer else int(role == "down")].append(name)
    paths = []
    for i, names in enumerate(shards):
        path = root / f"q2-0000{i + 1}-of-00003.gguf"
        writer = GGUFWriter(path, "qwen4exp")
        if i:
            writer.kv_data[0].pop("general.architecture", None)
        writer.add_uint16("split.no", i)
        writer.add_uint16("split.count", len(shards))
        writer.add_int32("split.tensors.count", len(tensors))
        for name in names:
            raw, kind = tensors[name]
            writer.add_tensor(name, raw, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(path)
    return paths, tensors


class UnslothQ2PackTests(unittest.TestCase):
    def test_native_roles_and_explicit_small_projection_conversion(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths, tensors = q2_shards(root)
            before = [path.read_bytes() for path in paths]
            out = root / "pack"
            rc, log = run_pack(paths[0], out)
            self.assertEqual(rc, 1, log)
            self.assertIn("use --compat-bf16", log)
            self.assertFalse((out / "native_experts.txt").exists())

            rc, log = run_pack(paths[0], out, "--compat-bf16")
            self.assertEqual(rc, 0, log)
            _, rows = iq_pack.read_index(out / "index.txt")
            dense = (out / "dense.bin").read_bytes()
            converted = {"blk.0.hc_attn_down.weight", "blk.1.ple_value.weight",
                         "blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight"}
            records = {r["name"]: r for r in json.loads((out / "conversions.json").read_text())["tensors"]}
            self.assertEqual(set(records), converted)
            self.assertEqual(records["blk.0.hc_attn_down.weight"]["shard"], paths[0].name)
            self.assertEqual(records["blk.1.ple_value.weight"]["shard"], paths[1].name)
            for name, (raw, kind) in tensors.items():
                if name.endswith("_exps.weight") or name == "per_layer_token_embd.weight":
                    self.assertNotIn(name, rows)
                elif name in converted:
                    self.assertEqual(rows[name][2], "4")
                    decoded = quants.dequantize(raw, kind).ravel()
                    bits = decoded.view(np.uint32)
                    expected = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
                    np.testing.assert_array_equal(stored(rows, dense, name, "<u2"), expected)
                    self.assertEqual((records[name]["src_type"], records[name]["dst_type"]), (kind.name, "BF16"))
                    self.assertEqual(records[name]["source_sha256"], hashlib.sha256(raw.tobytes()).hexdigest())
                else:
                    # The head, embedding, native dense matrices and Q8_0 PLE key stay in their GGUF.
                    self.assertEqual((rows[name][2], rows[name][4], rows[name][6]), ("0", "0", "0"), name)
                    self.assertEqual(rows[name][7:9], ["256", "4"], name)
            self.assertEqual(len(dense), len(converted) * 4 * 256 * 2)
            self.assertFalse((out / "experts.bin").exists())
            self.assertEqual([path.read_bytes() for path in paths], before)

    def test_expert_formats_and_per_role_shard_bytes_survive_packing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths, tensors = q2_shards(root)
            out = root / "pack"
            rc, log = run_pack(paths[0], out, "--compat-bf16", "--experts-bin")
            self.assertEqual(rc, 0, log)
            header, *layers = (out / "native_experts.txt").read_text().splitlines()
            self.assertTrue(header.startswith("# strata native experts v4:"), header)
            expected = bytearray()
            for layer, (line, gu_type) in enumerate(zip(layers, (Q.IQ2_XS, Q.IQ3_XXS))):
                fields = line.split()
                self.assertEqual([int(v) for v in fields[:3]], [layer, int(gu_type), int(Q.IQ4_NL)])
                self.assertEqual(int(fields[3]), len(expected))
                shard_names = fields[8].split(",")
                if len(shard_names) == 1:
                    shard_names *= 3
                roles = [tensors[f"blk.{layer}.ffn_{role}_exps.weight"][0].reshape(4, -1)
                         for role in ("gate", "up", "down")]
                self.assertEqual(int(fields[4]), sum(raw.shape[1] for raw in roles))
                for role, shard, offset in zip(roles, shard_names, fields[5:8]):
                    source = root / shard if shard else paths[0]
                    with source.open("rb") as handle:
                        handle.seek(int(offset))
                        self.assertEqual(handle.read(role.nbytes), role.tobytes())
                for expert in range(4):
                    for role in roles:
                        expected.extend(role[expert].tobytes())
            self.assertEqual(len(layers), 2)
            self.assertEqual(layers[0].split()[8], f",,{paths[1].name}")
            self.assertEqual(layers[1].split()[8], paths[2].name)
            self.assertEqual((out / "experts.bin").read_bytes(), bytes(expected))


if __name__ == "__main__":
    unittest.main()
