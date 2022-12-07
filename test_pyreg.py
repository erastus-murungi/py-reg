import re

from match import RegexMatcher
from simplify import simplify


def test_simply_maintains_simple_constructs():
    cases = [
        (r"a", "a"),
        (r"ab", "ab"),
        (r"ab|cd", "ab|cd"),
        (r"(ab)*", "(ab)*"),
        (r".", "."),
        (r"^", "^"),
        (r"$", "$"),
        (r"ABC[a-x]\d", "ABC[a-x]\\d"),
    ]
    for test_input, expected in cases:
        assert simplify(test_input) == expected, (test_input, expected)


# acquired from re2: https://github.com/google/re2/blob/main/re2/testing/search_test.cc


def _test_cases_suite(cases: list[tuple[str, str]]):
    for i, (pattern, text) in enumerate(cases):
        expected = [m.group(0) for m in re.finditer(pattern, text) if m.group(0) != ""]
        actual = [m.substr for m in RegexMatcher(pattern, text) if m.substr != ""]
        assert expected == actual, (i, pattern, text)


def test_interesting_cases():
    cases = [
        (r"a{,2}", "str{a{,2}}"),
        ("\\.\\^\\$\\\\", "str{.^$\\}"),
        (r"[a-zABC]", "cc{0x41-0x43 0x61-0x7a}"),
        (r"[^a]", "cc{0-0x60 0x62-0x10ffff}"),
        ("a*\\{", "cat{star{lit{a}}lit{{}}"),
    ]

    _test_cases_suite(cases)


def test_word_boundary_cases():
    cases = [
        ("\\bfoo\\b", "nofoo foo that"),
        ("a\\b", "faoa x"),
        ("\\bbar", "bar x"),
        ("\\bbar", "foo\nbar x"),
        ("bar\\b", "foobar"),
        ("bar\\b", "foobar\nxxx"),
        ("(foo|bar|[A-Z])\\b", "foo"),
        ("(foo|bar|[A-Z])\\b", "foo\n"),
        ("\\b", ""),
        ("\\b", "x"),
        ("\\b(foo|bar|[A-Z])", "foo"),
        ("\\b(foo|bar|[A-Z])\\b", "X"),
        ("\\b(foo|bar|[A-Z])\\b", "XY"),
        ("\\b(foo|bar|[A-Z])\\b", "bar"),
        ("\\b(foo|bar|[A-Z])\\b", "foo"),
        ("\\b(foo|bar|[A-Z])\\b", "foo\n"),
        ("\\b(foo|bar|[A-Z])\\b", "ffoo bbar N x"),
        ("\\b(fo|foo)\\b", "fo"),
        ("\\b(fo|foo)\\b", "foo"),
        ("\\b\\b", ""),
        ("\\b\\b", "x"),
        ("\\b$", ""),
        ("\\b$", "x"),
        ("\\b$", "y x"),
        ("\\b.$", "x"),
        ("^\\b(fo|foo)\\b", "fo"),
        ("^\\b(fo|foo)\\b", "foo"),
        ("^\\b", ""),
        ("^\\b", "x"),
        ("^\\b\\b", ""),
        ("^\\b\\b", "x"),
        ("^\\b$", ""),
        ("^\\b$", "x"),
        ("^\\b.$", "x"),
        ("^\\b.\\b$", "x"),
        ("^^^^^^^^\\b$$$$$$$", ""),
        ("^^^^^^^^\\b.$$$$$$", "x"),
        ("^^^^^^^^\\b$$$$$$$", "x"),
    ]

    _test_cases_suite(cases)


def test_non_word_boundary_cases():
    cases = [
        ("\\Bfoo\\B", "n foo xfoox that"),
        ("a\\B", "faoa x"),
        ("\\Bbar", "bar x"),
        ("\\Bbar", "foo\nbar x"),
        ("bar\\B", "foobar"),
        ("bar\\B", "foobar\nxxx"),
        ("(foo|bar|[A-Z])\\B", "foox"),
        ("(foo|bar|[A-Z])\\B", "foo\n"),
        ("\\B", ""),
        ("\\B", "x"),
        ("\\B(foo|bar|[A-Z])", "foo"),
        ("\\B(foo|bar|[A-Z])\\B", "xXy"),
        ("\\B(foo|bar|[A-Z])\\B", "XY"),
        ("\\B(foo|bar|[A-Z])\\B", "XYZ"),
        ("\\B(foo|bar|[A-Z])\\B", "abara"),
        ("\\B(foo|bar|[A-Z])\\B", "xfoo_"),
        ("\\B(foo|bar|[A-Z])\\B", "xfoo\n"),
        ("\\B(foo|bar|[A-Z])\\B", "foo bar vNx"),
        ("\\B(fo|foo)\\B", "xfoo"),
        ("\\B(foo|fo)\\B", "xfooo"),
        ("\\B\\B", ""),
        ("\\B\\B", "x"),
        ("\\B$", ""),
        ("\\B$", "x"),
        ("\\B$", "y x"),
        ("\\B.$", "x"),
        ("^\\B(fo|foo)\\B", "fo"),
        ("^\\B(fo|foo)\\B", "foo"),
        ("^\\B", ""),
        ("^\\B", "x"),
        ("^\\B\\B", ""),
        ("^\\B\\B", "x"),
        ("^\\B$", ""),
        ("^\\B$", "x"),
        ("^\\B.$", "x"),
        ("^\\B.\\B$", "x"),
        ("^^^^^^^^\\B$$$$$$$", ""),
        ("^^^^^^^^\\B.$$$$$$", "x"),
        ("^^^^^^^^\\B$$$$$$$", "x"),
    ]

    _test_cases_suite(cases)


def test_edges_cases():
    cases = [
        (r"a", "a"),
        (r"a", "zyzzyva"),
        (r"a+", "aa"),
        (r"(a+|b)+", "ab"),
        (r"ab|cd", "xabcdx"),
        (r"h.*od?", "hello\ngoodbye\n"),
        (r"h.*o", "hello\ngoodbye\n"),
        (r"h.*o", "goodbye\nhello\n"),
        (r"h.*o", "hello world"),
        (r"h.*o", "othello, world"),
        ("[^\\s\\S]", "aaaaaaa"),
        (r"a", "aaaaaaa"),
        (r"a*", "aaaaaaa"),
        (r"a*", ""),
        (r"ab|cd", "xabcdx"),
        (r"a", "cab"),
        (r"a*b", "cab"),
        (r"((((((((((((((((((((x))))))))))))))))))))", "x"),
        (r"[abcd]", "xxxabcdxxx"),
        (r"[^x]", "xxxabcdxxx"),
        (r"[abcd]+", "xxxabcdxxx"),
        (r"[^x]+", "xxxabcdxxx"),
        (r"(fo|foo)", "fo"),
        (r"(foo|fo)", "foo"),
        ("aa", "aA"),
        ("a", "Aa"),
        ("a", "A"),
        ("ABC", "abc"),
        ("abc", "XABCY"),
        ("ABC", "xabcy"),
        (r"foo|bar|[A-Z]", "foo"),
        (r"^(foo|bar|[A-Z])", "foo"),
        ("(foo|bar|[A-Z])$", "foo\n"),
        ("(foo|bar|[A-Z])$", "foo"),
        ("^(foo|bar|[A-Z])$", "foo\n"),
        ("^(foo|bar|[A-Z])$", "foo"),
        ("^(foo|bar|[A-Z])$", "bar"),
        ("^(foo|bar|[A-Z])$", "X"),
        ("^(foo|bar|[A-Z])$", "XY"),
        ("^(fo|foo)$", "fo"),
        ("^(fo|foo)$", "foo"),
        ("^^(fo|foo)$", "fo"),
        ("^^(fo|foo)$", "foo"),
        ("^$", ""),
        ("^$", "x"),
        ("^^$", ""),
        ("^$$", ""),
        ("^^$", "x"),
        ("^$$", "x"),
        ("^^$$", ""),
        ("^^$$", "x"),
        ("^^^^^^^^$$$$$$$$", ""),
        ("^", "x"),
        ("$", "x"),
        ("^$^$", ""),
        ("^$^", ""),
        ("$^$", ""),
        ("^$^$", "x"),
        ("^$^", "x"),
        ("$^$", "x"),
        ("^$^$", "x\ny"),
        ("^$^", "x\ny"),
        ("$^$", "x\ny"),
        ("^$^$", "x\n\ny"),
        ("^$^", "x\n\ny"),
        ("$^$", "x\n\ny"),
        ("^(foo\\$)$", "foo$bar"),
        ("(foo\\$)", "foo$bar"),
        ("^...$", "abc"),
        ("^abc", "abcdef"),
        ("^abc", "aabcdef"),
        ("^[ay]*[bx]+c", "abcdef"),
        ("^[ay]*[bx]+c", "aabcdef"),
        ("def$", "abcdef"),
        ("def$", "abcdeff"),
        ("d[ex][fy]$", "abcdef"),
        ("d[ex][fy]$", "abcdeff"),
        ("[dz][ex][fy]$", "abcdef"),
        ("[dz][ex][fy]$", "abcdeff"),
    ]

    _test_cases_suite(cases)
