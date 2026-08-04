from vision_tokenization.parsers.multilingual_recap import parse


def test_multilingual_recap_accepts_hyphenated_language_tags():
    text = (
        "lang=pt-br\n"
        "Legenda em portugues.\n"
        "lang=zh-cn\n"
        "中文说明。\n"
    )

    segments = parse(text)

    assert segments == [
        {"type": "image"},
        {"type": "text", "text": "Legenda em portugues.", "lang": "pt-br"},
        {"type": "text", "text": "中文说明。", "lang": "zh-cn"},
    ]
