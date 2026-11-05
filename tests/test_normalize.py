from agents.text_agent.normalize import normalize_text


def test_homoglyphs_and_zero_width_are_undone():
    attacked = "Th\u0435 qu\u0456ck br\u043e\u0301wn\u200b fox"      # Cyrillic е, і, о + combining mark, zero-width space
    clean, stats = normalize_text("The quick brown fox")
    out, s = normalize_text("Th\u0435 qu\u0456ck br\u043ewn\u200b fox")
    assert out == clean
    assert s["homoglyph_chars"] == 3 and s["zero_width_chars"] == 1 and s["tamper_ratio"] > 0
    assert stats["tamper_ratio"] == 0
    assert attacked  # sanity


def test_genuine_non_latin_text_is_not_mangled():
    russian = "Это обычный русский текст без подмены букв"
    out, stats = normalize_text(russian)
    assert out == russian and stats["homoglyph_chars"] == 0


def test_pasted_duplicates_collapse_to_one_copy():
    from agents.text_agent.normalize import normalize_text
    t = "This is a reasonably long sentence that should be repeated. It keeps going for a while."
    out, stats = normalize_text(t + "\n\n" + t + "\n\n" + t)
    assert out == t and stats["repeated_copies"] == 3
    assert normalize_text(t)[1]["repeated_copies"] == 1
