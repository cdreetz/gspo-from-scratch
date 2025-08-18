def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("###")[1].strip()
    except IndexError:
        return None
