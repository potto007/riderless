# 0005. Relocatable prebuilt worker bundles

- Status: accepted
- Date: 2026-09-22

## Context

Getting a first answer out of unridden took two compiles. The quick start
asked for CMake, a C++ toolchain and, for a GPU, the CUDA toolkit, then spent
about ten minutes building llama.cpp for one architecture and a minute more on
the worker. That is a reasonable price for someone who intends to change the
worker. It is an unreasonable price for someone deciding whether the idea is
worth ten minutes at all, and it rules out anyone who cannot install a CUDA
toolkit on the machine that has the GPU.

Publishing a binary was not possible as the build stood, because the worker
install was not portable in two independent ways.

The manifest recorded absolute paths. `build.json` carried
`"executable": "/home/someone/unridden/build/api-worker/build/unridden-worker"`
and a `runtime_dir` pointing at the llama.cpp base runtime somewhere else
entirely, and startup compared the configured worker against that absolute
string. Moving the directory, let alone unpacking it on another machine, failed
the check. The manifest also named the base build directory outright, which is
a path from the builder's machine sitting in an artifact.

The worker also had its backend directory compiled in.
`ggml_backend_load_all_from_path(UNRIDDEN_BACKEND_DIR)` used a string CMake
baked from `LLAMA_BUILD`, so the binary could only ever dlopen the ggml
backends out of the tree it was compiled against. The same absolute path was
in its run path, which additionally made the executable's sha256 depend on
where it was built.

## Decision

A worker install is a **bundle**: one directory that holds everything and names
nothing outside itself.

```
<dir>/build.json                the manifest
<dir>/build/unridden-worker    the executable
<dir>/runtime/*.so*             the libraries it links and dlopens
```

`ApiConfig`'s defaults already point at `build/api-worker/build/unridden-worker`
and `build/api-worker/build.json`, so a bundle unpacked into `build/api-worker`
needs no flags. The local build produces that exact shape - it now copies the
base runtime's libraries into the bundle instead of pointing at them - so a
compiled install and a downloaded one are the same thing.

- **Manifest schema 2.** `executable` and `runtime_dir` are relative to the
  manifest's own directory, and are resolved against it. An absolute path in a
  schema 2 manifest is refused, as is a relative path that escapes the bundle.
  `base_build`, which held the builder's path, is gone; `base_build_json_sha256`
  keeps the provenance it carried. Schema 1 manifests are still read with their
  absolute paths, so an install built before 0.2.0 keeps starting; they are
  simply not relocatable, and `pack` refuses them.
- **`--runtime-dir` replaces the compile-time constant.** The worker and the so1
  probe take the ggml backend directory as an argument. The API backend passes
  the directory it just finished hashing, so the worker loads backends from
  exactly the files the manifest vouched for.
- **`$ORIGIN/../runtime` as the executable's run path**, and `LD_LIBRARY_PATH`
  as the mechanism that actually covers the load. Both, because neither alone
  is enough. The run path removes the build machine's path from the binary and
  resolves the worker's own `DT_NEEDED` entries. It does not reach further: the
  llama.cpp libraries carry their own `DT_RUNPATH`, which suppresses run-path
  inheritance for everything they pull in, and rewriting theirs would change
  the sha256 the manifest pins. `LD_LIBRARY_PATH`, which the backend already
  exported and which outranks `DT_RUNPATH`, is what resolves those transitive
  loads and the dlopened backends. Measured on an unpacked bundle: without
  `LD_LIBRARY_PATH` the loader found `libggml-base.so.0` through a stale
  absolute `DT_RUNPATH`; with it, every library resolved inside the bundle.
- **CUDA bundles carry the CUDA runtime.** `libcudart`, `libcublas` and
  `libcublasLt` are copied next to the ggml CUDA backend and hashed with the
  rest, the way llama.cpp's own `cudart-*` tarballs do, so the only host
  requirement is the NVIDIA driver. `libcuda.so.1` is the driver and is
  deliberately not copied.
- **Published, verified, fetched.** A tagged release builds `cuda13`, `cuda12`
  and `cpu` bundles in NVIDIA's devel containers with `GGML_NATIVE=OFF` and a
  broad architecture set, attaches them with a `SHA256SUMS` file, and attests
  them with `actions/attest-build-provenance`. `unridden-api worker fetch`
  picks a flavor from the installed driver, checks the download against
  `SHA256SUMS`, runs `gh attestation verify` when `gh` is there, and unpacks.

## What is still verified at startup

Nothing was relaxed. Before the child is allowed to start, the API still:

- resolves the manifest's paths and refuses a worker other than the one the
  manifest names;
- re-hashes the executable against `executable_sha256`;
- re-hashes every file in `runtime_sha256` inside the bundle's runtime
  directory, and checks the whole inventory against `runtime_bundle_sha256`;
- computes the model's sha256, and enforces it when one is configured;
- refuses a manifest that does not say `generated_tokens: 0`,
  `callbacks_enabled: false` and `execution_mode: "full"`;
- warns, but does not refuse, when the llama.cpp revision is not the tested one
  ([ADR 0003](0003-tested-default-revision-instead-of-a-hard-pin.md)).

A fetched bundle earns nothing a compiled one does not. What the download adds
on top is the `SHA256SUMS` check against the release and GitHub's build
provenance; what it removes is the compiler, not a check.

## Alternatives rejected

- **Keep absolute paths and rewrite the manifest on unpack.** A fetch step
  would edit `build.json` to name wherever it landed. It works, and it means
  the artifact people verify is not the artifact that runs: any hash over the
  manifest stops being meaningful, and a hand-moved directory breaks again.
- **Patch `DT_RUNPATH` in the shipped libraries with patchelf.** Would make the
  bundle work with no `LD_LIBRARY_PATH` at all. It rewrites the very files
  whose sha256 the base build recorded, so either the manifest is regenerated
  after tampering - which is exactly the shape of the attack the hashes exist
  to catch - or the check fails. Not worth a variable that was already set.
- **Statically link the backends.** llama.cpp's backend registry is built
  around dlopen, and a static worker would give up the per-backend selection
  and the `libggml-cuda.so` discovery that goes with it.
- **A PyPI wheel carrying the binary.** A 450 MB manylinux wheel per flavor,
  and pip has no way to choose between them from the driver version. A release
  asset plus an explicit fetch command says out loud which build it picked.
- **Build the published bundles for one architecture.** Faster, and useless to
  anyone whose GPU is not a 5090.

## Consequences

- **A published binary is not promised to be bit-identical to the numbers in
  `docs/results/`.** Those were measured on a native build for a single
  architecture. The release bundles are built with `GGML_NATIVE=OFF` and
  `CMAKE_CUDA_ARCHITECTURES=80;86;89;90;120`, which is a different compile.
  The README says so, and the determinism claim stays what it always was:
  deterministic per configuration, not across configurations.
- **The CUDA bundles are large.** `libcublasLt` alone is about 500 MB
  uncompressed; a cuda13 bundle is roughly 450 MB compressed. That is the price
  of a user needing only a driver, and it is what llama.cpp's own releases pay.
- **This project now redistributes NVIDIA's CUDA runtime libraries** in its
  release artifacts, under the CUDA EULA. Nothing is vendored into the
  repository; the libraries are copied at build time from the CUDA container.
  `NOTICE` records it.
- **A pre-0.2.0 worker directory keeps working but cannot be packed.** Schema 1
  is read; rebuilding is what produces a relocatable one.
- **The worker gained a required argument.** Anything that launched
  `unridden-worker` by hand must now pass `--runtime-dir`. The so1 probe and
  its benchmark driver did, and were changed.

## What would reverse this

- Upstream making the ggml backends discoverable without a directory argument
  at all, or llama.cpp publishing per-revision shared-library releases we could
  consume directly, would remove most of what the base runtime build exists for.
- Evidence that a broad-architecture published build moves answers enough to
  matter would force the honest response: publish per-architecture bundles, or
  stop publishing binaries and go back to the source build being the only
  supported path.
- A supply-chain requirement stricter than GitHub's build provenance - a
  reproducible build that a third party can reproduce byte for byte - would
  mean pinning the container digests and the toolchain, which this workflow
  does not do today.
