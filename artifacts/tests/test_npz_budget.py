"""Allocation and decompression guards, using only small synthetic archives."""

import io
import struct
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from artifacts.janus_artifact.legacy_npz import (
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    MAX_NPY_HEADER_BYTES,
    _inspect_zip,
    inspect_prefill_rank_npz,
)
from artifacts.janus_artifact.schema import ValidationError


def npy_header(shape, *, descr="<u2", body=b"\x01\x00", version=(1, 0)):
    header = repr({"descr": descr, "fortran_order": False, "shape": shape})
    encoding = "utf8" if version == (3, 0) else "latin1"
    header_bytes = (header + "\n").encode(encoding)
    length_format = "<H" if version == (1, 0) else "<I"
    return (
        b"\x93NUMPY"
        + bytes(version)
        + struct.pack(length_format, len(header_bytes))
        + header_bytes
        + body
    )


class NpzBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.payload = Path(self.directory.name) / "synthetic.npz"

    def write_archive(self, attn, *, top_k=None, extra=None):
        if top_k is None:
            top_k = npy_header(())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.payload, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("attn_rank.npy", attn)
                archive.writestr("top_k.npy", top_k)
                if extra is not None:
                    archive.writestr(*extra)

    def assert_rejected_before_crc_and_load(self, message):
        with (
            mock.patch.object(zipfile.ZipFile, "testzip") as crc,
            mock.patch.object(np, "load") as load,
            self.assertRaisesRegex(ValidationError, message),
        ):
            inspect_prefill_rank_npz(self.payload)
        crc.assert_not_called()
        load.assert_not_called()

    def test_duplicate_member_names_rejected_before_crc(self):
        array = npy_header((1, 1, 1, 1))
        self.write_archive(array, extra=("attn_rank.npy", array))
        self.assert_rejected_before_crc_and_load("duplicate member names")

    def test_unexpected_member_rejected_before_crc(self):
        self.write_archive(npy_header((1, 1, 1, 1)), extra=("extra.npy", b"small"))
        self.assert_rejected_before_crc_and_load("keys must equal")

    def test_giant_directory_member_rejected_without_decompression(self):
        members = [
            zipfile.ZipInfo("attn_rank.npy"),
            zipfile.ZipInfo("top_k.npy"),
        ]
        members[0].file_size = DEFAULT_MAX_UNCOMPRESSED_BYTES + 1
        members[1].file_size = 2
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.infolist.return_value = members
        with mock.patch.object(zipfile, "ZipFile", return_value=archive):
            with self.assertRaisesRegex(ValidationError, "uncompressed-size limit"):
                _inspect_zip(self.payload, DEFAULT_MAX_UNCOMPRESSED_BYTES)
        archive.open.assert_not_called()
        archive.testzip.assert_not_called()

    def test_directory_total_budget_rejected_without_decompression(self):
        members = [
            zipfile.ZipInfo("attn_rank.npy"),
            zipfile.ZipInfo("top_k.npy"),
        ]
        for member in members:
            member.file_size = 60
        archive = mock.MagicMock()
        archive.__enter__.return_value = archive
        archive.infolist.return_value = members
        with mock.patch.object(zipfile, "ZipFile", return_value=archive):
            with self.assertRaisesRegex(ValidationError, "uncompressed-size limit"):
                _inspect_zip(self.payload, 100)
        archive.open.assert_not_called()
        archive.testzip.assert_not_called()

    def test_tiny_body_with_giant_shape_rejected_before_allocation(self):
        self.write_archive(npy_header((1, 1, 1, 10**12)))
        self.assert_rejected_before_crc_and_load("declared array exceeds")

    def test_declared_bytes_must_match_small_body(self):
        self.write_archive(npy_header((1, 2, 2, 2)))
        self.assert_rejected_before_crc_and_load("declared array bytes")

    def test_object_dtype_rejected_from_header_without_pickle(self):
        self.write_archive(npy_header((1, 1, 1, 1), descr="|O", body=b"small"))
        self.assert_rejected_before_crc_and_load("object dtype")

    def test_header_budget_rejected_without_reading_giant_header(self):
        attn = b"\x93NUMPY\x02\x00" + struct.pack("<I", MAX_NPY_HEADER_BYTES + 1)
        self.write_archive(attn)
        self.assert_rejected_before_crc_and_load("NPY header-size limit")

    def test_invalid_shape_rejected_from_header(self):
        for shape in ((1, 1, 1, True), (1, 1, 1, 0), (1, 1, 1, 1.5)):
            with self.subTest(shape=shape):
                self.write_archive(npy_header(shape))
                self.assert_rejected_before_crc_and_load("shape|positive")

    def test_top_k_must_be_integer_scalar_before_load(self):
        for top_k in (
            npy_header((1,)),
            npy_header((), descr="<f8", body=b"\x00" * 8),
        ):
            with self.subTest(top_k=top_k):
                self.write_archive(npy_header((1, 1, 1, 1)), top_k=top_k)
                self.assert_rejected_before_crc_and_load("integer scalar|integer dtype")

    def test_valid_numeric_archive_still_uses_crc_and_pickle_disabled(self):
        for version in ((1, 0), (2, 0), (3, 0)):
            with self.subTest(version=version):
                self.write_archive(npy_header((1, 1, 1, 1), version=version))
                with (
                    mock.patch.object(zipfile.ZipFile, "testzip", return_value=None) as crc,
                    mock.patch.object(np, "load", wraps=np.load) as load,
                ):
                    result = inspect_prefill_rank_npz(self.payload)
                crc.assert_called_once()
                load.assert_called_once_with(self.payload, allow_pickle=False)
                self.assertEqual(result["arrays"][0]["shape"], [1, 1, 1, 1])
                self.assertEqual(result["arrays"][0]["nbytes"], 2)

    def test_crc_failure_rejected_before_numpy_load(self):
        self.write_archive(npy_header((1, 1, 1, 1)))
        with (
            mock.patch.object(zipfile.ZipFile, "testzip", return_value="attn_rank.npy") as crc,
            mock.patch.object(np, "load") as load,
            self.assertRaisesRegex(ValidationError, "CRC verification failed"),
        ):
            inspect_prefill_rank_npz(self.payload)
        crc.assert_called_once()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
