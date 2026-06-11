from types import SimpleNamespace

import pytest

from sglang.srt.speculative.dflash_worker import (
    _resolve_dflash_draft_kv_cache_dtype,
)


def _args(kv_cache_dtype="fp8_e4m3", speculative_draft_kv_cache_dtype=None):
    return SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        speculative_draft_kv_cache_dtype=speculative_draft_kv_cache_dtype,
    )


def test_dflash_fa4_draft_defaults_to_bfloat16():
    assert _resolve_dflash_draft_kv_cache_dtype(_args(), "fa4") == "bfloat16"


def test_dflash_fa4_draft_rejects_explicit_fp8_kv():
    with pytest.raises(ValueError, match="fa4 attention requires"):
        _resolve_dflash_draft_kv_cache_dtype(
            _args(speculative_draft_kv_cache_dtype="fp8_e4m3"), "fa4"
        )


def test_dflash_non_fa4_draft_inherits_target_kv_dtype():
    assert _resolve_dflash_draft_kv_cache_dtype(_args(), "flashinfer") == "fp8_e4m3"


def test_dflash_non_fa4_draft_uses_explicit_kv_dtype():
    assert (
        _resolve_dflash_draft_kv_cache_dtype(
            _args(speculative_draft_kv_cache_dtype="bfloat16"), "flashinfer"
        )
        == "bfloat16"
    )
