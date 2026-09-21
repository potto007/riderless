"""Native stdio lifecycle tests with a disposable CPU-only protocol child."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from riderless.api.backend import (
    BackendExecutionError,
    BackendProtocolError,
    BackendRequestError,
    BackendUnavailableError,
)
from riderless.api.compiler import CompiledBatch, compile_request
from riderless.api.native.build import TESTED_LLAMA_REVISION
from riderless.api.native_backend import NativeBackend
from riderless.api.schema import BackendProfile, DecisionRequest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_install(
    tmp_path: Path,
    mode: str = "normal",
    revision: str = TESTED_LLAMA_REVISION,
) -> tuple[Path, Path, Path]:
    worker = tmp_path / "worker"
    worker.write_text(
        """#!/usr/bin/env python3
import argparse, hashlib, json, sys, time
p=argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--model-sha256')
p.add_argument('--runtime-sha256'); p.add_argument('--context', type=int)
p.add_argument('--batch', type=int); p.add_argument('--ubatch', type=int)
p.add_argument('--threads', type=int); p.add_argument('--max-questions', type=int)
p.add_argument('--gpu', action='store_true')
p.add_argument('--batched', action='store_true')
p.add_argument('--batched-context', type=int, default=0)
a=p.parse_args()
labels=['A','B','C']; token_ids=[11,12,13]
print(json.dumps({'type':'hello','protocol_version':3,
 'model_id':'local-gemma-riderless-v1','model_name':'fixture',
 'model_sha256':a.model_sha256,'runtime_sha256':a.runtime_sha256,
 'labels':labels,'label_token_ids':token_ids,'context_size':a.context,
 'batch_size':a.batch,'ubatch_size':a.ubatch,'threads':a.threads,
 'max_questions':a.max_questions,
 'batched_mode':a.batched,
 'batched_context':a.batched_context if a.batched else 0,
 'generated_tokens':0,
 'callbacks_enabled':False,'execution_mode':'full'}), flush=True)
print('fixture native diagnostic', file=sys.stderr, flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if 'MODE_FLOOD' in __file__:
  sys.stderr.write('e' * 5000); sys.stderr.flush(); time.sleep(60)
 if 'MODE_HANG' in __file__: time.sleep(60)
 if 'MODE_CORRUPT' in __file__: print('{', flush=True); continue
 if 'MODE_INTERNAL' in __file__:
  print(json.dumps({'type':'error','id':request['id'],'code':'invalid_request',
   'reason':'internal'}), flush=True); continue
 if 'MODE_CONTROL' in __file__:
  print(json.dumps({'type':'error','id':request['id'],'code':'invalid_request',
   'reason':'control_tokens'}), flush=True); continue
 if 'MODE_OVERSIZE' in __file__: print('x' * 5000, flush=True); continue
 rows=[]
 mode='batched' if a.batched else 'sequential'
 siblings=len(request['questions']) if a.batched else 1
 for i,q in enumerate(request['questions']):
  count=len(q['labels'])
  rows.append({'id':q['id'],'label_logits':[float(x) for x in range(count)],
   'label_token_ids':token_ids[:count],'allowed_label_mass':0.5,
   'full_vocabulary_argmax':{'token_id':12,'logit':3.0},
   'prompt_sha256':hashlib.sha256(q['messages'][0]['content'].encode()).hexdigest(),
   'prompt_tokens':10+i,'processed_tokens':10+i,'reused_tokens':0,
   'cache_cleared':True,
   'evaluation_mode':mode,'batch_sequences':siblings,
   'timing_ms':1.0})
 print(json.dumps({'type':'result','id':request['id'],
  'model_sha256':a.model_sha256,'runtime_sha256':a.runtime_sha256,
  'generated_tokens':0,'callbacks_enabled':False,'execution_mode':'full',
  'batched_fallback':None,
  'questions':rows}), flush=True)
"""
    )
    if mode != "normal":
        renamed = tmp_path / f"worker-MODE_{mode.upper()}"
        worker.rename(renamed)
        worker = renamed
    worker.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake model")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    library = runtime / "libfixture.so"
    library.write_bytes(b"runtime")
    checksums = {library.name: _digest(library)}
    manifest = tmp_path / "build.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "llama_revision": revision,
                "runtime_dir": str(runtime),
                "runtime_sha256": checksums,
                "runtime_bundle_sha256": hashlib.sha256(
                    json.dumps(
                        checksums, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "executable": str(worker),
                "executable_sha256": _digest(worker),
                "generated_tokens": 0,
                "callbacks_enabled": False,
                "execution_mode": "full",
            }
        )
    )
    return worker, model, manifest


def _batch(profile: BackendProfile) -> CompiledBatch:
    request = DecisionRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"left": None, "right": None},
                }
            },
        }
    )
    return compile_request(request, profile)


@pytest.mark.asyncio
async def test_native_backend_owns_one_correlated_child(tmp_path: Path) -> None:
    worker, model, manifest = _fake_install(tmp_path)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )

    profile = await backend.start()
    owned_pid = backend.pid
    result = await backend.evaluate(_batch(profile), timeout=2.0)
    for _ in range(200):
        if backend.stderr_tail:
            break
        await asyncio.sleep(0.01)
    diagnostic_tail = backend.stderr_tail
    await backend.close()

    assert isinstance(owned_pid, int) and owned_pid > 0
    assert result.questions[0].id == "q"
    assert result.id
    assert diagnostic_tail == ("fixture native diagnostic",)
    assert backend.pid is None
    assert backend.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("corrupt", BackendProtocolError),
        ("oversize", BackendProtocolError),
        ("hang", TimeoutError),
        ("flood", TimeoutError),
    ],
)
async def test_protocol_corruption_or_timeout_reaps_owned_child(
    tmp_path: Path, mode: str, error: type[BaseException]
) -> None:
    worker, model, manifest = _fake_install(tmp_path, mode)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        max_response_bytes=2048,
        startup_timeout=2.0,
    )
    profile = await backend.start()
    owned_pid = backend.pid

    with pytest.raises(error):
        await backend.evaluate(
            _batch(profile), timeout=0.3 if mode in {"hang", "flood"} else 2.0
        )

    assert backend.pid is None
    assert backend.ready is False
    assert owned_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(owned_pid, 0)


@pytest.mark.asyncio
async def test_batched_opt_in_reaches_the_child_and_is_held_to_the_handshake(
    tmp_path: Path,
) -> None:
    worker, model, manifest = _fake_install(tmp_path)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
        batched=True,
        batched_context=4096,
    )

    profile = await backend.start()
    result = await backend.evaluate(_batch(profile), timeout=2.0)
    await backend.close()

    assert (profile.batched_mode, profile.batched_context) == (True, 4096)
    assert result.questions[0].evaluation_mode == "batched"
    assert result.batched_fallback is None

    # The child is started without the flag, so its handshake disagrees.
    mismatched = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )
    mismatched.batched = True
    with pytest.raises(BackendUnavailableError):
        await mismatched.start()


@pytest.mark.asyncio
async def test_sequential_default_never_asks_the_child_for_batched_work(
    tmp_path: Path,
) -> None:
    worker, model, manifest = _fake_install(tmp_path)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )

    profile = await backend.start()
    result = await backend.evaluate(_batch(profile), timeout=2.0)
    await backend.close()

    assert (profile.batched_mode, profile.batched_context) == (False, 0)
    assert result.questions[0].evaluation_mode == "sequential"
    assert result.questions[0].batch_sequences == 1


@pytest.mark.asyncio
async def test_model_hash_pin_refuses_a_different_model(tmp_path: Path) -> None:
    worker, model, manifest = _fake_install(tmp_path)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
        expected_model_sha256="0" * 64,
    )

    with pytest.raises(BackendUnavailableError):
        await backend.start()

    assert backend.pid is None
    assert backend.ready is False


@pytest.mark.asyncio
async def test_control_token_rejection_keeps_the_child_and_names_its_reason(
    tmp_path: Path,
) -> None:
    worker, model, manifest = _fake_install(tmp_path, "control")
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
        expected_model_sha256=_digest(model),
    )
    profile = await backend.start()

    with pytest.raises(BackendRequestError) as caught:
        await backend.evaluate(_batch(profile), timeout=2.0)
    still_ready = backend.ready
    await backend.close()

    assert caught.value.reason == "control_tokens"
    assert still_ready is True


@pytest.mark.asyncio
async def test_tested_revision_starts_and_says_so_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    worker, model, manifest = _fake_install(tmp_path)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )

    with caplog.at_level("WARNING"):
        profile = await backend.start()
    await backend.close()

    assert profile.llama_revision == TESTED_LLAMA_REVISION
    assert profile.tested_revision is True
    assert caplog.records == []


@pytest.mark.asyncio
async def test_untested_revision_starts_but_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = "0" * 40
    worker, model, manifest = _fake_install(tmp_path, revision=other)
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )

    with caplog.at_level("WARNING"):
        profile = await backend.start()
    result = await backend.evaluate(_batch(profile), timeout=2.0)
    await backend.close()

    # An untested revision is a provenance warning, never a refusal.
    assert profile.llama_revision == other
    assert profile.tested_revision is False
    assert result.questions[0].id == "q"
    assert len(caplog.records) == 1
    warning = caplog.records[0].getMessage()
    assert other in warning and TESTED_LLAMA_REVISION in warning


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["runtime", "executable", "revision"])
async def test_corrupted_manifest_is_still_refused(
    tmp_path: Path, corruption: str
) -> None:
    worker, model, manifest = _fake_install(tmp_path)
    recorded = json.loads(manifest.read_text())
    if corruption == "runtime":
        # The library on disk is no longer the one the manifest vouched for.
        (Path(recorded["runtime_dir"]) / "libfixture.so").write_bytes(b"tampered")
    elif corruption == "executable":
        recorded["executable_sha256"] = "0" * 64
        manifest.write_text(json.dumps(recorded))
    else:
        del recorded["llama_revision"]
        manifest.write_text(json.dumps(recorded))

    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )

    with pytest.raises(BackendUnavailableError):
        await backend.start()

    assert backend.pid is None
    assert backend.ready is False


@pytest.mark.asyncio
async def test_internal_preflight_disagreement_is_not_a_client_error(
    tmp_path: Path,
) -> None:
    worker, model, manifest = _fake_install(tmp_path, "internal")
    backend = NativeBackend(
        worker_path=worker,
        model_path=model,
        manifest_path=manifest,
        startup_timeout=2.0,
    )
    profile = await backend.start()

    with pytest.raises(BackendExecutionError):
        await backend.evaluate(_batch(profile), timeout=2.0)
    still_ready = backend.ready
    await backend.close()

    assert still_ready is True
