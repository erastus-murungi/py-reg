from match import Regexp


def test_perl_character_classes():
    cases = [
        ("\\d", "[0-9]"),
        ("\\s", "[ \t\n\r\x0b\x0c]"),
        ("\\w", "[0-9A-Z_a-z]"),
        ("\\D", "[^0-9]"),
        ("\\S", "[^ \t\n\r\x0b\x0c]"),
        ("\\W", "[^0-9A-Z_a-z]"),
    ]

    for regex, expected in cases:
        assert Regexp(regex).recover() == expected, (
            regex,
            expected,
            Regexp(regex).recover(),
        )
