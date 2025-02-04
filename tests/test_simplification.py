import pytest

from reg.parser import RegexParser


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("\\d", "[0-9]"),
        ("\\s", "[ \t\n\r\x0b\x0c]"),
        ("\\w", "[0-9A-Z_a-z]"),
        ("\\D", "[^0-9]"),
        ("\\S", "[^ \t\n\r\x0b\x0c]"),
        ("\\W", "[^0-9A-Z_a-z]"),
    ],
)
def test_perl_character_classes(pattern, expected):
    recovered = RegexParser(pattern).root.to_string()
    assert recovered == expected, f"{pattern=}, {expected=}, {recovered=}"
