# API Stability and Versioning

This document defines the stability guarantees and versioning strategy for the Micro-Swarm DLL API (`micro_swarm_api.h`).

## Versioning Strategy

- **Semantic versioning** for the API surface: `MAJOR.MINOR.PATCH`
  - **MAJOR**: Breaking ABI or behavior changes.
  - **MINOR**: Backward-compatible additions (new functions/struct fields).
  - **PATCH**: Bug fixes, no API changes.

## ABI Rules

- The C-API uses **__cdecl** calling convention on Windows.
- Structs are POD, `repr(C)`, default alignment (no packing pragmas).
- Function names remain stable; if a function must change, a new name is introduced.

## Struct Evolution

- New fields are **appended** to structs.
- Existing fields are never reordered or removed.
- Callers should zero-initialize structs and set required fields explicitly.

## Error Semantics

- `int` return values:
  - `> 0`: success
  - `0`: failure (invalid handle, invalid inputs, buffer too small, etc.)

## Deprecation Policy

- Deprecated functions remain for at least one **MAJOR** release.
- Deprecated entries are documented in `API_Stability.md` and marked in the header.

## Compatibility Matrix

- **Windows**: `micro_swarm.dll`
- **Linux**: `libmicro_swarm.so`
- Same API, same semantics.

## Change Log Location

- Changes to the API surface are recorded in `API_Stability.md` under a dated section.

## Runtime Version Query

- `ms_get_api_version(int* major, int* minor, int* patch)` returns the runtime API version for dynamic loaders.

## Header Version Macros

- `MS_API_VERSION_MAJOR`
- `MS_API_VERSION_MINOR`
- `MS_API_VERSION_PATCH`
