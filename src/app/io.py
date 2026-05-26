import os

from src.app.config import (
    DRAFT_REVIEWED_MODEL_PATH,
    OUTPUT_DIR,
    REVIEW_PATH,
)


def load_text_file(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    return None


def save_text_file(path: str, text: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def load_review_file():
    return load_text_file(REVIEW_PATH)


def save_review_file(text: str):
    save_text_file(REVIEW_PATH, text)


def append_latest_answer_to_review_draft(answer: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = load_text_file(DRAFT_REVIEWED_MODEL_PATH) or ""
    addition = f"\n\n## Added from paper Q&A\n\n{answer.strip()}\n"

    with open(DRAFT_REVIEWED_MODEL_PATH, "w", encoding="utf-8") as file:
        file.write(existing.rstrip() + addition)
