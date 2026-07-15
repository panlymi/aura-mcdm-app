from __future__ import annotations

import tomllib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from mcdm.uploads import (
    MAX_ALTERNATIVES,
    MAX_CELLS,
    MAX_CRITERIA,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_MEBIBYTES,
    MAX_XLSX_UNCOMPRESSED_BYTES,
    content_fingerprint,
    load_decision_matrix,
    validate_matrix_limits,
    validate_upload_size,
)
from mcdm.validation import MCDMValidationError


ROOT = Path(__file__).resolve().parents[1]


def _example_matrix() -> pd.DataFrame:
    data = pd.DataFrame(
        {"C1": [1.5, 2.5], "C2": [3.0, 4.0]},
        index=["A1", "A2"],
    )
    data.index.name = "Alternative"
    return data


def _xlsx_bytes(data: pd.DataFrame) -> bytes:
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        data.to_excel(writer)
    return workbook.getvalue()


def test_content_fingerprint_is_exact_deterministic_sha256():
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    assert content_fingerprint(b"abc") == expected
    assert content_fingerprint(bytearray(b"abc")) == expected
    assert content_fingerprint(b"abd") != expected


def test_load_csv_accepts_case_insensitive_suffix():
    expected = _example_matrix()

    loaded = load_decision_matrix("DECISION-MATRIX.CSV", expected.to_csv().encode("utf-8"))

    pd.testing.assert_frame_equal(loaded, expected, check_dtype=False)


def test_load_xlsx_accepts_case_insensitive_suffix():
    expected = _example_matrix()

    loaded = load_decision_matrix("DECISION-MATRIX.XLSX", _xlsx_bytes(expected))

    pd.testing.assert_frame_equal(loaded, expected, check_dtype=False)


def test_csv_preflight_respects_quoted_commas_and_newlines():
    content = (
        'Alternative,"Criterion, one"\r\n'
        '"Alternative\r\none",1.5\r\n'
    ).encode("utf-8")

    loaded = load_decision_matrix("quoted.csv", content)

    assert loaded.columns.tolist() == ["Criterion, one"]
    assert loaded.index.tolist() == ["Alternative\r\none"]
    assert loaded.iloc[0, 0] == 1.5


@pytest.mark.parametrize(
    ("filename", "content", "format_name"),
    [
        ("broken.CSV", b"\xff\xfe\x00not-utf-8", "CSV"),
        ("broken-quotes.CSV", b'Alternative,C1\n"A1,1\n', "CSV"),
        ("broken.XLSX", b"not an Excel workbook", "XLSX"),
    ],
)
def test_malformed_uploads_raise_friendly_parser_errors(filename, content, format_name):
    with pytest.raises(MCDMValidationError, match=rf"Could not read .* as {format_name}") as exc:
        load_decision_matrix(filename, content)

    assert exc.value.__cause__ is not None
    assert "Traceback" not in str(exc.value)


def test_empty_csv_raises_friendly_parser_error():
    with pytest.raises(MCDMValidationError, match="Could not read .* as CSV"):
        load_decision_matrix("empty.csv", b"")


def test_unsupported_suffix_is_rejected_before_parsing():
    with pytest.raises(MCDMValidationError, match=r"\.csv or \.xlsx"):
        load_decision_matrix("matrix.xls", b"anything")


def test_upload_size_limit_accepts_boundary_and_rejects_one_extra_byte():
    at_limit = b"x" * MAX_UPLOAD_BYTES
    validate_upload_size(at_limit)

    with pytest.raises(MCDMValidationError, match=f"maximum is {MAX_UPLOAD_MB} MB"):
        load_decision_matrix("matrix.csv", at_limit + b"x")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (pd.DataFrame(columns=["C1"]), "at least one alternative"),
        (pd.DataFrame(index=["A1"]), "at least one criterion"),
    ],
)
def test_matrix_limits_reject_empty_dimensions(data, message):
    with pytest.raises(MCDMValidationError, match=message):
        validate_matrix_limits(data)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"Alternative,C1\n", "at least one alternative"),
        (b"Alternative\nA1\n", "at least one criterion"),
    ],
)
def test_csv_upload_rejects_empty_dimensions(content, message):
    with pytest.raises(MCDMValidationError, match=message):
        load_decision_matrix("empty-dimension.csv", content)


def test_csv_preflight_rejects_huge_header_before_pandas(monkeypatch):
    criteria = [f"C{i}" for i in range(10_000)]
    content = (",".join(["Alternative", *criteria]) + "\n").encode("utf-8")

    def fail_if_called(*args, **kwargs):
        pytest.fail("pandas must not parse a CSV that fails bounded preflight")

    monkeypatch.setattr(pd, "read_csv", fail_if_called)
    with pytest.raises(MCDMValidationError, match="criteria"):
        load_decision_matrix("huge-header.csv", content)


def test_csv_preflight_rejects_excessive_rows_before_pandas(monkeypatch):
    rows = ["Alternative,C1", *(f"A{i},{i}" for i in range(MAX_ALTERNATIVES + 1))]
    content = ("\n".join(rows) + "\n").encode("utf-8")

    def fail_if_called(*args, **kwargs):
        pytest.fail("pandas must not parse a CSV that fails bounded preflight")

    monkeypatch.setattr(pd, "read_csv", fail_if_called)
    with pytest.raises(MCDMValidationError, match="alternatives"):
        load_decision_matrix("too-many-rows.csv", content)


def test_xlsx_preflight_rejects_excessive_dimensions_before_pandas(monkeypatch):
    oversized = pd.DataFrame(
        [[1] * (MAX_CRITERIA + 1)],
        columns=[f"C{i}" for i in range(MAX_CRITERIA + 1)],
        index=["A1"],
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("pandas must not parse an XLSX that fails bounded preflight")

    monkeypatch.setattr(pd, "read_excel", fail_if_called)
    with pytest.raises(MCDMValidationError, match="criteria"):
        load_decision_matrix("too-wide.xlsx", _xlsx_bytes(oversized))


def test_xlsx_expansion_cap_allows_normal_maximum_matrix():
    boundary = pd.DataFrame(
        1.0,
        index=[f"A{i}" for i in range(MAX_ALTERNATIVES)],
        columns=[f"C{i}" for i in range(MAX_CRITERIA)],
    )
    workbook = _xlsx_bytes(boundary)
    with ZipFile(BytesIO(workbook)) as archive:
        expanded_size = sum(member.file_size for member in archive.infolist())

    assert expanded_size < MAX_XLSX_UNCOMPRESSED_BYTES
    loaded = load_decision_matrix("maximum-normal.xlsx", workbook)
    assert loaded.shape == (MAX_ALTERNATIVES, MAX_CRITERIA)


def test_xlsx_preflight_rejects_compressed_expansion_bomb_before_pandas(monkeypatch):
    archive_buffer = BytesIO(_xlsx_bytes(_example_matrix()))
    with ZipFile(
        archive_buffer,
        mode="a",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            "xl/oversized-payload.bin",
            b"0" * (MAX_XLSX_UNCOMPRESSED_BYTES + 1),
        )
    compressed_workbook = archive_buffer.getvalue()
    assert len(compressed_workbook) < MAX_UPLOAD_BYTES

    def fail_if_called(*args, **kwargs):
        pytest.fail("pandas must not parse an XLSX expansion bomb")

    monkeypatch.setattr(pd, "read_excel", fail_if_called)
    with pytest.raises(MCDMValidationError, match="XLSX archive expands"):
        load_decision_matrix("compressed-bomb.xlsx", compressed_workbook)


def test_matrix_limits_accept_exact_combined_boundary():
    boundary = pd.DataFrame(index=range(MAX_ALTERNATIVES), columns=range(MAX_CRITERIA))

    validate_matrix_limits(boundary)
    assert boundary.size == MAX_CELLS


def test_matrix_limits_reject_too_many_alternatives():
    data = pd.DataFrame(index=range(MAX_ALTERNATIVES + 1), columns=["C1"])

    with pytest.raises(MCDMValidationError, match="alternatives"):
        validate_matrix_limits(data)


def test_matrix_limits_reject_too_many_criteria():
    data = pd.DataFrame(index=["A1"], columns=range(MAX_CRITERIA + 1))

    with pytest.raises(MCDMValidationError, match="criteria"):
        validate_matrix_limits(data)


def test_matrix_limits_enforce_combined_cell_ceiling():
    data = pd.DataFrame(
        index=range(MAX_ALTERNATIVES + 1),
        columns=range(MAX_CRITERIA),
    )

    with pytest.raises(MCDMValidationError, match=rf"maximum is {MAX_CELLS:,}"):
        validate_matrix_limits(data)


def test_streamlit_transport_upload_limit_matches_app_limit():
    with (ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    assert config["server"]["maxUploadSize"] == MAX_UPLOAD_MEBIBYTES
