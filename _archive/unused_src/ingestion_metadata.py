from datetime import datetime
import hashlib
import json


def hash_file(path: str) -> str:
    digest = hashlib.md5()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def extraction_metadata(
    *,
    source_pdf: str,
    page=None,
    bbox=None,
    chunk_id=None,
    object_type: str = "text",
    extraction_method: str = "text_layer",
    confidence=None,
    requires_review: bool = False,
    cache_path=None,
    extra: dict | None = None,
) -> dict:
    metadata = {
        "source_pdf": source_pdf,
        "page": page,
        "bbox": bbox,
        "chunk_id": chunk_id,
        "object_type": object_type,
        "extraction_method": extraction_method,
        "confidence": confidence,
        "requires_review": requires_review,
        "cache_path": cache_path,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if extra:
        metadata.update(extra)

    return metadata


def cache_key(*parts) -> str:
    normalized = json.dumps(parts, sort_keys=True, default=str)
    return hash_text(normalized)
