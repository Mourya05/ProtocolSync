"""
core/parser.py
──────────────────────────────────────────────────────────────────────────────
Ingests and normalises OpenAPI 3.0 / 3.1 specifications (JSON or YAML) into a
strongly-typed internal schema model for the Protocol-Sync fuzzing harness.

Public surface
--------------
  parse_file(path)             → ParsedAPI
  parse_string(text, fmt)      → ParsedAPI
  ParsedAPI, EndpointSpec, FieldSpec  (exported dataclasses)

Design decisions
----------------
* $ref resolution is implemented as a recursive document walk (no external
  libraries) to guarantee determinism and avoid circular-ref surprises from
  jsonschema's resolution graph.
* Only local document references (``#/…``) are supported in Phase 1.  External
  file refs raise ValueError with a clear message.
* All returned dataclasses use ``frozen=True`` (EndpointSpec, FieldSpec) or are
  plain mutable dataclasses (ParsedAPI) for ease of downstream mutation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldSpec:
    """
    Strongly-typed descriptor for a single API parameter or body property.

    Attributes
    ----------
    location : ``"path"`` | ``"query"`` | ``"header"`` | ``"cookie"`` | ``"body"``
    type     : JSON Schema primitive (``"string"``, ``"integer"``, ``"number"``,
               ``"boolean"``, ``"array"``, ``"object"``)
    """

    name: str
    location: Literal["path", "query", "header", "cookie", "body"]
    type: str                               # JSON Schema primitive
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None
    enum: tuple[Any, ...] | None = None
    default: Any = None
    description: str | None = None
    items_type: str | None = None           # element type when ``type == "array"``
    # Nested properties for object types — tuple for hashability
    properties: tuple["FieldSpec", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EndpointSpec:
    """
    Normalised, immutable representation of a single API endpoint.

    All collection fields use tuples so the dataclass is hashable and safe to
    use as dictionary keys or in sets.
    """

    path: str
    method: str                                         # lowercase: get, post, put …
    operation_id: str | None
    summary: str | None
    path_params: tuple[FieldSpec, ...]
    query_params: tuple[FieldSpec, ...]
    header_params: tuple[FieldSpec, ...]
    request_body_schema: dict[str, Any] | None          # fully-resolved JSON Schema dict
    request_body_required: bool
    body_fields: tuple[FieldSpec, ...]                  # flat list of body properties
    response_schemas: dict[str, dict[str, Any]]         # "200" / "201" / … → schema dict


@dataclass
class ParsedAPI:
    """Top-level result object returned by :func:`parse_file` / :func:`parse_string`."""

    title: str
    version: str
    base_url: str
    endpoints: list[EndpointSpec]

    # ── Convenience ────────────────────────────────────────────────────────

    def endpoint(self, path: str, method: str) -> EndpointSpec | None:
        """Return the :class:`EndpointSpec` matching *path* and *method*, or ``None``."""
        target = method.lower()
        return next(
            (ep for ep in self.endpoints if ep.path == path and ep.method == target),
            None,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ParsedAPI(title={self.title!r}, version={self.version!r}, "
            f"base_url={self.base_url!r}, endpoints={len(self.endpoints)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# $ref resolver  (local document only — Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class RefResolver:
    """
    Resolves JSON Reference (``$ref``) pointers within a *single* OpenAPI document.

    Raises
    ------
    ValueError
        If a ref points outside the current document (external file refs).
    KeyError
        If the ref target cannot be located within the document.
    RecursionError
        If the resolution depth exceeds :attr:`_MAX_DEPTH` (circular ref guard).
    """

    _MAX_DEPTH: int = 50

    def __init__(self, document: dict[str, Any]) -> None:
        self._doc = document

    def resolve(self, schema: Any, _depth: int = 0) -> Any:  # noqa: ANN001
        """
        Recursively resolve every ``$ref`` in *schema* and return a clean copy.

        Non-dict / non-list values are returned as-is.
        """
        if _depth > self._MAX_DEPTH:
            raise RecursionError(
                "$ref resolution exceeded depth limit — possible circular reference in spec."
            )

        if isinstance(schema, dict):
            if "$ref" in schema:
                return self.resolve(self._follow_ref(schema["$ref"]), _depth + 1)
            return {k: self.resolve(v, _depth + 1) for k, v in schema.items()}

        if isinstance(schema, list):
            return [self.resolve(item, _depth + 1) for item in schema]

        return schema

    # ── Internal ────────────────────────────────────────────────────────────

    def _follow_ref(self, ref: str) -> Any:
        """Navigate the document tree using a local JSON Pointer (``#/a/b/c``)."""
        if not ref.startswith("#/"):
            raise ValueError(
                f"External $ref is not supported in Phase 1: {ref!r}. "
                "Only local document references (#/…) are resolved."
            )

        parts = ref[2:].split("/")   # strip "#/"
        node: Any = self._doc

        for raw_part in parts:
            # RFC 6901 JSON Pointer escape sequences
            part = raw_part.replace("~1", "/").replace("~0", "~")

            if not isinstance(node, dict):
                raise TypeError(
                    f"Cannot traverse $ref {ref!r}: expected a mapping at "
                    f"segment {part!r}, got {type(node).__name__}."
                )
            if part not in node:
                raise KeyError(
                    f"$ref target not found: {ref!r} (missing key {part!r})."
                )
            node = node[part]

        return node


# ─────────────────────────────────────────────────────────────────────────────
# Schema → FieldSpec conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _schema_to_fields(
    schema: dict[str, Any],
    *,
    location: Literal["path", "query", "header", "cookie", "body"] = "body",
) -> tuple[FieldSpec, ...]:
    """
    Convert a JSON Schema object's ``properties`` map into :class:`FieldSpec` tuples.

    Handles nested ``object`` types recursively and records ``items.type`` for arrays.
    Falls back gracefully when the schema is empty or lacks a ``properties`` key.
    """
    if not schema or "properties" not in schema:
        return ()

    required_set: set[str] = set(schema.get("required", []))
    specs: list[FieldSpec] = []

    for name, prop in schema["properties"].items():
        raw_type: str = prop.get("type", "string")
        items_type: str | None = None
        nested: tuple[FieldSpec, ...] = ()

        if raw_type == "array" and isinstance(prop.get("items"), dict):
            items_type = prop["items"].get("type", "object")

        if raw_type == "object" and "properties" in prop:
            nested = _schema_to_fields(prop, location=location)

        # Support both ``minimum`` and ``exclusiveMinimum`` (OAS 3.0 / 3.1)
        minimum: float | None = (
            prop["minimum"] if "minimum" in prop
            else prop.get("exclusiveMinimum")
        )
        maximum: float | None = (
            prop["maximum"] if "maximum" in prop
            else prop.get("exclusiveMaximum")
        )

        enum_raw = prop.get("enum")
        specs.append(
            FieldSpec(
                name=name,
                location=location,
                type=raw_type,
                required=name in required_set,
                minimum=minimum,
                maximum=maximum,
                pattern=prop.get("pattern"),
                enum=tuple(enum_raw) if enum_raw is not None else None,
                default=prop.get("default"),
                description=prop.get("description"),
                items_type=items_type,
                properties=nested,
            )
        )

    return tuple(specs)


def _parameter_to_field_spec(param: dict[str, Any]) -> FieldSpec:
    """Convert an OpenAPI *parameter* object into a :class:`FieldSpec`."""
    schema: dict[str, Any] = param.get("schema", {})
    raw_type: str = schema.get("type", "string")

    minimum: float | None = (
        schema["minimum"] if "minimum" in schema
        else schema.get("exclusiveMinimum")
    )
    maximum: float | None = (
        schema["maximum"] if "maximum" in schema
        else schema.get("exclusiveMaximum")
    )

    enum_raw = schema.get("enum")
    loc: str = param.get("in", "query")

    return FieldSpec(
        name=param["name"],
        location=loc,  # type: ignore[arg-type]
        type=raw_type,
        required=param.get("required", False),
        minimum=minimum,
        maximum=maximum,
        pattern=schema.get("pattern"),
        enum=tuple(enum_raw) if enum_raw is not None else None,
        default=schema.get("default"),
        description=param.get("description"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_VERSIONS: tuple[str, ...] = ("3.0", "3.1")
_HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
)


class OpenAPIParser:
    """
    Parse an OpenAPI 3.0 / 3.1 document dict into a :class:`ParsedAPI` model.

    Raises
    ------
    ValueError
        If the document is structurally invalid or the version is unsupported.
    KeyError
        If a ``$ref`` target cannot be resolved within the document.
    RecursionError
        If ``$ref`` resolution exceeds the maximum depth (circular reference guard).
    """

    def __init__(self, document: dict[str, Any]) -> None:
        self._raw = document
        self._resolver = RefResolver(document)
        self._validate_root()

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate_root(self) -> None:
        if "openapi" not in self._raw:
            raise ValueError(
                "Invalid OpenAPI document: missing required 'openapi' version field."
            )
        version = str(self._raw["openapi"])
        if not any(version.startswith(v) for v in _SUPPORTED_VERSIONS):
            raise ValueError(
                f"Unsupported OpenAPI version: {version!r}. "
                f"Phase 1 supports {_SUPPORTED_VERSIONS}."
            )
        if "info" not in self._raw:
            raise ValueError("Invalid OpenAPI document: missing required 'info' object.")
        if "paths" not in self._raw:
            raise ValueError("Invalid OpenAPI document: missing required 'paths' object.")

    # ── Public ──────────────────────────────────────────────────────────────

    def parse(self) -> ParsedAPI:
        """Parse the document and return a fully-populated :class:`ParsedAPI` instance."""
        info = self._raw.get("info", {})
        servers: list[dict[str, Any]] = self._raw.get("servers", [])
        base_url: str = servers[0].get("url", "http://localhost") if servers else "http://localhost"

        return ParsedAPI(
            title=info.get("title", "Unknown"),
            version=info.get("version", "0.0.0"),
            base_url=base_url,
            endpoints=self._parse_paths(),
        )

    # ── Path parsing ────────────────────────────────────────────────────────

    def _parse_paths(self) -> list[EndpointSpec]:
        endpoints: list[EndpointSpec] = []

        for path, path_item_raw in self._raw.get("paths", {}).items():
            path_item: dict[str, Any] = self._resolver.resolve(path_item_raw)
            # Path-level parameters are shared across all operations
            shared_params: list[dict[str, Any]] = path_item.get("parameters", [])

            for method in _HTTP_METHODS:
                if method not in path_item:
                    continue
                operation: dict[str, Any] = self._resolver.resolve(path_item[method])
                endpoints.append(
                    self._parse_operation(path, method, operation, shared_params)
                )

        return endpoints

    def _parse_operation(
        self,
        path: str,
        method: str,
        operation: dict[str, Any],
        shared_params: list[dict[str, Any]],
    ) -> EndpointSpec:
        # Merge path-level and operation-level params; operation wins on collision
        merged = self._merge_params(shared_params, operation.get("parameters", []))

        path_params: list[FieldSpec] = []
        query_params: list[FieldSpec] = []
        header_params: list[FieldSpec] = []

        for raw_param in merged:
            param = self._resolver.resolve(raw_param)
            fs = _parameter_to_field_spec(param)
            loc = param.get("in", "query")
            if loc == "path":
                path_params.append(fs)
            elif loc == "query":
                query_params.append(fs)
            elif loc == "header":
                header_params.append(fs)

        # ── Request body ─────────────────────────────────────────────────
        body_schema: dict[str, Any] | None = None
        body_required: bool = False
        body_fields: tuple[FieldSpec, ...] = ()

        if "requestBody" in operation:
            rb = self._resolver.resolve(operation["requestBody"])
            body_required = rb.get("required", False)
            content: dict[str, Any] = rb.get("content", {})

            # Prefer application/json, then any other declared content type
            chosen_ct: str | None = next(
                (ct for ct in ("application/json", "*/*") if ct in content),
                next(iter(content), None),
            )
            if chosen_ct is not None:
                raw_schema = content[chosen_ct].get("schema", {})
                body_schema = self._resolver.resolve(raw_schema)
                if body_schema:
                    body_fields = _schema_to_fields(body_schema)

        # ── Response schemas ──────────────────────────────────────────────
        response_schemas: dict[str, dict[str, Any]] = {}
        for status_code, resp_raw in operation.get("responses", {}).items():
            resp = self._resolver.resolve(resp_raw)
            json_content = resp.get("content", {}).get("application/json", {})
            if "schema" in json_content:
                response_schemas[str(status_code)] = self._resolver.resolve(
                    json_content["schema"]
                )

        return EndpointSpec(
            path=path,
            method=method,
            operation_id=operation.get("operationId"),
            summary=operation.get("summary"),
            path_params=tuple(path_params),
            query_params=tuple(query_params),
            header_params=tuple(header_params),
            request_body_schema=body_schema,
            request_body_required=body_required,
            body_fields=body_fields,
            response_schemas=response_schemas,
        )

    @staticmethod
    def _merge_params(
        shared: list[dict[str, Any]],
        operation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge path-level and operation-level parameters.

        Operation-level parameters override shared ones when the ``(name, in)``
        key matches (per OpenAPI specification §4.8.9).
        """
        index: dict[tuple[str, str], dict[str, Any]] = {
            (p.get("name", ""), p.get("in", "")): p for p in shared
        }
        for p in operation:
            index[(p.get("name", ""), p.get("in", ""))] = p
        return list(index.values())


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_file(path: str | Path) -> ParsedAPI:
    """
    Parse an OpenAPI spec from *path* on disk.

    The format (JSON vs. YAML) is inferred from the file extension.
    ``.yaml`` / ``.yml`` files are parsed as YAML; all others as JSON.

    Raises
    ------
    FileNotFoundError : if *path* does not exist.
    ValueError        : if the document cannot be parsed or is structurally invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OpenAPI spec file not found: {p}")

    text = p.read_text(encoding="utf-8")
    fmt: Literal["json", "yaml"] = (
        "yaml" if p.suffix.lower() in {".yaml", ".yml"} else "json"
    )
    return parse_string(text, fmt=fmt)


def parse_string(
    text: str,
    *,
    fmt: Literal["json", "yaml"] = "json",
) -> ParsedAPI:
    """
    Parse an OpenAPI spec from a raw string.

    Parameters
    ----------
    text : Raw document text.
    fmt  : ``"json"`` (default) or ``"yaml"``.

    Raises
    ------
    ValueError
        If *text* is empty, cannot be parsed, or is not a valid OpenAPI 3.x spec.
    """
    if not text.strip():
        raise ValueError("Empty OpenAPI spec string provided.")

    try:
        document: Any = json.loads(text) if fmt == "json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Failed to parse {fmt.upper()} document: {exc}"
        ) from exc

    if not isinstance(document, dict):
        raise ValueError(
            f"OpenAPI spec must be a JSON object / YAML mapping; "
            f"got {type(document).__name__}."
        )

    return OpenAPIParser(document).parse()
