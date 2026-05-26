def text_layer_is_weak(text: str, min_length: int = 40) -> bool:
    cleaned = (text or "").strip()
    return len(cleaned) < min_length
