import re
from random import randint, random, seed

import pytest

from reg.parser import RegexpParsingError
from reg.pike_vm import RegexPikeVM


def get_compiled_vm(pattern):
    return RegexPikeVM(pattern)


def _test_case_no_groups(pattern: str, text: str) -> None:
    expected = [m.group(0) for m in re.finditer(pattern, text)]
    actual = [m.group(0) for m in get_compiled_vm(pattern).finditer(text)]
    assert expected == actual, (pattern, text)


def _test_case(pattern: str, text: str) -> None:
    _test_case_no_groups(pattern, text)

    expected_groups = [m.groups() for m in re.finditer(pattern, text)]
    actual_groups = [m.groups() for m in get_compiled_vm(pattern).finditer(text)]
    for expected_group, actual_group in zip(expected_groups, actual_groups):
        assert expected_group == actual_group, (pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("ab{0,}bc", "abbbbc"),
        ("ab{1,}bc", "abbbbc"),
        ("ab{1,3}bc", "abbbbc"),
        ("ab{3,4}bc", "abbbbc"),
        ("ab{4,5}bc", "abbbbc"),
        ("ab{0,1}bc", "abc"),
        ("ab{0,1}c", "abc"),
        ("^", "abc"),
        ("$", "abc"),
        ("ab{1,}bc", "abq"),
        ("a{1,}b{1,}c", "aabbabc"),
        ("(a+|b){0,}", "ab"),
        ("(a+|b){1,}", "ab"),
        ("(a+|b){0,1}", "ab"),
        ("([abc])*d", "abbbcd"),
        ("([abc])*bcd", "abcd"),
    ],
)
def test_repetition(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        (r"a{,2}", "str{a{,2}}"),
        ("\\.\\^\\$\\\\", "str{.^$\\}"),
        (r"[a-zABC]", "cc{0x41-0x43 0x61-0x7a}"),
        (r"[^a]", "cc{0-0x60 0x62-0x10ffff}"),
        ("a*\\{", "cat{star{lit{a}}lit{{}}"),
        (r"a", "a"),
        (r"ab", "ab"),
        (r"ab|cd", "ab|cd"),
        (r"(ab)*", "(ab)*"),
        (r".", "."),
        (r"^", "^"),
        (r"$", "$"),
        (r"ABC[a-x]\d", "ABC[a-x]\\d"),
    ],
)
def test_interesting_cases(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
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
    ],
)
def test_word_boundary_cases(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
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
    ],
)
def test_non_word_boundary_cases(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
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
        ("", ""),  # empty string
    ],
)
def test_edge_cases(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [  # test common prefix
        ("Python|Perl", "Perl"),  # Alternation
        ("(Python|Perl)", "Perl"),  # Grouped alternation
        ("Python|Perl|Tcl", "Perl"),  # Alternation
        ("(Python|Perl|Tcl)", "Perl"),  # Grouped alternation
        ("([0a-z][a-z0-9]*,)+", "a5,b7,c9,"),  # Disable the fastmap optimization
        ("([a-z][a-z0-9]*,)+", "a5,b7,c9,"),  # A few sets
        ("Python", "Python"),  # Simple text literal
        (".*Python", "Python"),  # Bad text literal
        (".*Python.*", "Python"),  # Worse text literal
        (".*(Python)", "Python"),  # Bad text literal with grouping
    ],
)
def test_python_benchmark(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        (")", ""),
        ("a[]b", "-"),
        ("a[", "-"),
        ("a\\", "-"),
        ("abc)", "-"),
        ("(abc", "-"),
        (")(", "-"),
        ("a[b-a]", "-"),  # wrong order
        ("*a", "-"),
        ("(*)b", "-"),
        ("a**", ""),
        (r"^*", ""),
        ("(?i)a[b-a]", "-"),
        ("(?i)a[]b", "-"),
        ("(?i)a[", "-"),
        ("(?i)*a", "-"),
        ("(?i)(*)b", "-"),
        ("(?i)a\\", "-"),
        ("(?i)abc)", "-"),
        ("(?i)(abc", "-"),
        ("(?i)a**", "-"),
        ("(?i))(", "-"),
    ],
)
def test_raises_exception(pattern, text):
    with pytest.raises(re.error):
        _ = [m.group(0) for m in re.finditer(pattern, text) if m.group(0) != ""]
    with pytest.raises((RegexpParsingError, ValueError)):
        _ = [
            m.substr for m in get_compiled_vm(pattern).finditer(text) if m.substr != ""
        ]


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("abc", "xbc"),
        ("abc", "axc"),
        ("abc", "abx"),
        ("[k]", "ab"),
        ("multiple words of text", "uh-uh"),
        ("(bc+d$|ef*g.|h?i(j|k))", "effg"),
        ("(bc+d$|ef*g.|h?i(j|k))", "bcdd"),
        ("a[bcd]+dcdcde", "adcdcde"),
        ("^(ab|cd)e", "abcde"),
        ("$b", "b"),
        ("([abc]*)x", "abc"),
        ("a[^-b]c", "a-c"),
        ("a[^\\]b]c", "a]c"),
        ("z\\B", "xyz"),
        ("\\Bx", "xyz"),
        ("\\Ba\\B", "a-"),
        ("\\Ba\\B", "-a"),
        ("\\Ba\\B", "-a-"),
        ("\\By\\B", "xy"),
        ("\\By\\B", "yz"),
        ("\\by\\b", "xy"),
        ("\\by\\b", "yz"),
        ("\\by\\b", "xyz"),
        ("x\\b", "xyz"),
        ("a[b-d]e", "abd"),
        ("abc", ""),
        ("a.*c", "axyzd"),
        ("a[bc]d", "abc"),
        ("ab+bc", "abc"),
        ("ab+bc", "abq"),
        ("ab?bc", "abbbbc"),
        ("^abc$", "abcc"),
        ("^abc$", "aabc"),
        ("a[^bc]d", "abd"),
        (r"^a*?$", "foo"),
        (r"a[^>]*?b", "a>b"),
    ],
)
def test_failures(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("abc", "abc"),
        ("abc", "xabcy"),
        ("abc", "ababc"),
        ("ab*c", "abc"),
        ("ab*bc", "abc"),
        ("ab*bc", "abbc"),
        ("ab*bc", "abbbbc"),
        ("ab+bc", "abbc"),
        ("ab+bc", "abbbbc"),
        ("ab?bc", "abbc"),
        ("ab?bc", "abc"),
        ("ab?c", "abc"),
        ("^abc$", "abc"),
        ("^abc", "abcc"),
        ("abc$", "aabc"),
        ("^", "abc"),
        ("$", "abc"),
        ("a.c", "abc"),
        ("a.c", "axc"),
        ("a.*c", "axyzc"),
        ("a[bc]d", "abd"),
        ("a[b-d]e", "ace"),
        ("a[b-d]", "aac"),
        ("a[-b]", "a-"),
        ("a[\\-b]", "a-"),
        ("a\\]", "a\\]"),
        (r"a[\]]b", "a\\]b"),
        ("a[\\]]b", "a]b"),
        ("a[^bc]d", "aed"),
        ("a[^-b]c", "adc"),
        ("a[^\\]b]c", "adc"),
        ("\\ba\\b", "a-"),
        ("\\ba\\b", "-a"),
        ("\\ba\\b", "-a-"),
        ("x\\B", "xyz"),
        ("\\Bz", "xyz"),
        ("\\By\\b", "xy"),
        ("\\by\\B", "yz"),
        ("\\By\\B", "xyz"),
        ("ab|cd", "abc"),
        ("ab|cd", "abcd"),
        ("()ef", "def"),
        ("a\\(b", "a(b"),
        ("a\\(*b", "ab"),
        ("a\\(*b", "a((b"),
        ("a\\\\b", "a\\b"),
        ("((a))", "abc"),
        ("(a)b(c)", "abc"),
        ("a+b+c", "aabbabc"),
        ("(a+|b)*", "ab"),
        ("(a+|b)+", "ab"),
        ("(a+|b)?", "ab"),
        ("[^ab]*", "cde"),
        ("a*", ""),
        ("a|b|c|d|e", "e"),
        ("(a|b|c|d|e)f", "ef"),
        ("abcd*efg", "abcdefg"),
        ("ab*", "xabyabbbz"),
        ("ab*", "xayabbbz"),
        ("(ab|cd)e", "abcde"),
        ("[abhgefdc]ij", "hij"),
        ("(abc|)ef", "abcdef"),
        ("(a|b)c*d", "abcd"),
        ("(ab|ab*)bc", "abc"),
        ("a([bc]*)c*", "abc"),
        ("a([bc]*)(c*d)", "abcd"),
        ("a([bc]+)(c*d)", "abcd"),
        ("a([bc]*)(c+d)", "abcd"),
        ("a[bcd]*dcdcde", "adcdcde"),
        ("(ab|a)b*c", "abc"),
        ("((a)(b)c)(d)", "abcd"),
        ("[a-zA-Z_][a-zA-Z0-9_]*", "alpha"),
        ("^a(bc+|b[eh])g|.h$", "abh"),
        ("(bc+d$|ef*g.|h?i(j|k))", "effgz"),
        ("(bc+d$|ef*g.|h?i(j|k))", "ij"),
        ("(bc+d$|ef*g.|h?i(j|k))", "reffgz"),
        ("(((((((((a)))))))))", "a"),
        ("multiple words", "multiple words, yeah"),
        ("(.*)c(.*)", "abcde"),
        ("\\((.*), (.*)\\)", "(a, b)"),
        ("a[-]?c", "ac"),
        ("(a)(b)c|ab", "ab"),
        ("(a)+x", "aaax"),
        ("([ac])+x", "aacx"),
        ("([^/]*/)*sub1/", "d:msgs/tdir/sub1/trial/away.cpp"),
        ("([^.]*)\\.([^:]*):[T ]+(.*)", "track1.title:TBlah blah blah"),
        ("([^N]*N)+", "abNNxyzN"),
        ("([^N]*N)+", "abNNxyz"),
        ("([abc]*)x", "abcx"),
        ("([xyz]*)x", "abcx"),
        ("(a)+b|aac", "aac"),
    ],
)
def test_more_python_re_implementation_cases(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("a.+?c", "abcabc"),
        ("a.*?c", "abcabc"),
        ("a.{0,5}?c", "abcabc"),
        ("a{2,3}?", "aaaaa"),
    ],
)
def test_greedy_vs_lazy(pattern, text):
    _test_case(pattern, text)


seed(10)


@pytest.mark.parametrize(
    "pattern, text",
    [(r"0[xX](_?[0-9a-fA-F])+", hex(randint(0, 100000))) for _ in range(20)],
)
def test_hex(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [(r"(0(_?0)*|[1-9](_?[0-9])*)", str(randint(0, 100000))) for _ in range(20)],
)
def test_dec(pattern, text):
    _test_case(pattern, text)


def group(*choices):
    return "(" + "|".join(choices) + ")"


def maybe(*choices):
    return group(*choices) + "?"


exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
pointfloat = group(
    r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
) + maybe(exponent)


@pytest.mark.parametrize(
    "pattern, text",
    [(pointfloat, str(random())) for _ in range(20)]
    + [(pointfloat, f"{random():.2E}") for _ in range(20)]
    + [(pointfloat, f"{-random():.2E}") for _ in range(20)]
    + [(pointfloat, f"{-random()}") for _ in range(20)]
    + [(pointfloat, f"{10 + random()}") for _ in range(20)],
)
def test_point_float(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize("pattern, text", [("🇺🇸+", "🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸")])
def test_unicode_simple(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(?i)abc", "ABC"),
        ("(?i)abc", "XBC"),
        ("(?i)abc", "AXC"),
        ("(?i)abc", "ABX"),
        ("(?i)abc", "XABCY"),
        ("(?i)abc", "XABCY"),
        ("(?i)ab*c", "ABC"),
        ("(?i)abc", "ABABC"),
        ("(?i)ab{1,}?bc", "ABBBBC"),
        ("(?i)ab*c", "ABC"),
        ("(?i)ab*bc", "ABC"),
        ("(?i)ab*bc", "ABBC"),
        ("(?i)ab*?bc", "ABBBBC"),
        ("(?i)ab{0,}?bc", "ABBBBC"),
        ("(?i)ab+?bc", "ABBC"),
        ("(?i)ab+bc", "ABC"),
        ("(?i)ab+bc", "ABQ"),
        ("(?i)ab{1,}bc", "ABQ"),
        ("(?i)ab+bc", "ABBBBC"),
        ("(?i)ab{1,}?bc", "ABBBBC"),
        ("(?i)ab{1,3}?bc", "ABBBBC"),
        ("(?i)ab{3,4}?bc", "ABBBBC"),
        ("(?i)ab{4,5}?bc", "ABBBBC"),
        ("(?i)ab??bc", "ABBC"),
        ("(?i)ab??bc", "ABC"),
        ("(?i)ab{0,1}?bc", "ABC"),
        ("(?i)ab??bc", "ABBBBC"),
        ("(?i)ab??c", "ABC"),
        ("(?i)ab{0,1}?c", "ABC"),
        ("(?i)^abc$", "ABC"),
        ("(?i)^abc$", "ABCC"),
        ("(?i)^abc", "ABCC"),
        ("(?i)^abc$", "AABC"),
        ("(?i)abc$", "AABC"),
        ("(?i)^", "ABC"),
        ("(?i)$", "ABC"),
        ("(?i)a.c", "ABC"),
        ("(?i)a.c", "AXC"),
        ("(?i)a.*?c", "AXYZC"),
        ("(?i)ab*c", "ABC"),
        ("(?i)ab*bc", "ABC"),
        ("(?i)ab*bc", "ABBC"),
        ("(?i)ab*?bc", "ABBBBC"),
        ("(?i)ab{0,}?bc", "ABBBBC"),
        ("(?i)ab+?bc", "ABBC"),
        ("(?i)ab+bc", "ABC"),
        ("(?i)ab+bc", "ABQ"),
        ("(?i)ab{1,}bc", "ABQ"),
        ("(?i)ab+bc", "ABBBBC"),
        ("(?i)ab{1,}?bc", "ABBBBC"),
        ("(?i)ab{1,3}?bc", "ABBBBC"),
        ("(?i)ab{3,4}?bc", "ABBBBC"),
        ("(?i)ab{4,5}?bc", "ABBBBC"),
        ("(?i)ab??bc", "ABBC"),
        ("(?i)ab??bc", "ABC"),
        ("(?i)ab{0,1}?bc", "ABC"),
        ("(?i)ab??bc", "ABBBBC"),
        ("(?i)ab??c", "ABC"),
        ("(?i)ab{0,1}?c", "ABC"),
        ("(?i)^abc$", "ABC"),
        ("(?i)^abc$", "ABCC"),
        ("(?i)^abc", "ABCC"),
        ("(?i)^abc$", "AABC"),
        ("(?i)abc$", "AABC"),
        ("(?i)^", "ABC"),
        ("(?i)$", "ABC"),
        ("(?i)a.c", "ABC"),
        ("(?i)a.c", "AXC"),
        ("(?i)a.*?c", "AXYZC"),
        ("(?i)a.*c", "AXYZD"),
        ("(?i)a[bc]d", "ABC"),
        ("(?i)a[bc]d", "ABD"),
        ("(?i)a[b-d]e", "ABD"),
        ("(?i)a[b-d]e", "ACE"),
        ("(?i)a[b-d]", "AAC"),
        ("(?i)a[-b]", "A-"),
        ("(?i)a[b-]", "A-"),
        ("(?i)[^ab]*", "CDE"),
        ("(?i)abc", ""),
        ("(?i)a*", ""),
        ("(?i)([abc])*d", "ABBBCD"),
        ("(?i)([abc])*bcd", "ABCD"),
        ("(?i)a|b|c|d|e", "E"),
        ("(?i)(a|b|c|d|e)f", "EF"),
        ("(?i)abcd*efg", "ABCDEFG"),
        ("(?i)ab*", "XABYABBBZ"),
        ("(?i)ab*", "XAYABBBZ"),
        ("(?i)(ab|cd)e", "ABCDE"),
        ("(?i)[abhgefdc]ij", "HIJ"),
        ("(?i)^(ab|cd)e", "ABCDE"),
        ("(?i)(abc|)ef", "ABCDEF"),
        ("(?i)(a|b)c*d", "ABCD"),
        ("(?i)(ab|ab*)bc", "ABC"),
        ("(?i)a([bc]*)c*", "ABC"),
        ("(?i)a([bc]*)(c*d)", "ABCD"),
        ("(?i)a([bc]+)(c*d)", "ABCD"),
        ("(?i)a([bc]*)(c+d)", "ABCD"),
        ("(?i)a[bcd]*dcdcde", "ADCDCDE"),
        ("(?i)a[bcd]+dcdcde", "ADCDCDE"),
        ("(?i)(ab|a)b*c", "ABC"),
        ("(?i)((a)(b)c)(d)", "ABCD"),
        ("(?i)a\\]", "A]"),
        ("(?i)a[\\]]b", "A]B"),
        ("(?i)a[^bc]d", "AED"),
        ("(?i)a[^bc]d", "ABD"),
        ("(?i)a[^-b]c", "ADC"),
        ("(?i)a[^-b]c", "A-C"),
        ("(?i)a[^\\]b]c", "A]C"),
        ("(?i)a[^\\]b]c", "ADC"),
        ("(?i)ab|cd", "ABC"),
        ("(?i)ab|cd", "ABCD"),
        ("(?i)()ef", "DEF"),
        ("(?i)a\\(b", "A(B"),
        ("(?i)a\\(*b", "AB"),
        ("(?i)a\\(*b", "A((B"),
        ("(?i)a\\\\b", "A\\B"),
        ("(?i)a.+?c", "ABCABC"),
        ("(?i)a.*?c", "ABCABC"),
        ("(?i)a.{0,5}?c", "ABCABC"),
        ("(?i)(a+|b)*", "AB"),
        ("(?i)(a+|b){0,}", "AB"),
        ("(?i)(a+|b)+", "AB"),
        ("(?i)(a+|b){1,}", "AB"),
        ("(?i)(a+|b)?", "AB"),
        ("(?i)(a+|b){0,1}", "AB"),
        ("(?i)$b", "B"),
        ("(?i)(((((((((a)))))))))", "A"),
        ("(?i)(?:(?:(?:(?:(?:(?:(?:(?:(?:(a))))))))))", "A"),
        ("(?i)(?:(?:(?:(?:(?:(?:(?:(?:(?:(a|b|c))))))))))", "C"),
        ("(?i)multiple words of text", "UH-UH"),
        ("(?i)multiple words", "MULTIPLE WORDS, YEAH"),
        ("(?i)(.*)c(.*)", "ABCDE"),
        ("(?i)\\((.*), (.*)\\)", "(A, B)"),
        ("(?i)[k]", "AB"),
        ("(?i)abcd", "ABCD"),
        ("(?i)a(bc)d", "ABCDE"),
        ("(?i)a[-]?c", "AC"),
        ("(?i)((a))", "ABC"),
        ("(?i)(a)b(c)", "ABC"),
        ("(?i)a+b+c", "AABBABC"),
        ("(?i)a{1,}b{1,}c", "AABBABC"),
        ("(?i)^a(bc+|b[eh])g|.h$", "ABH"),
        ("(?i)(bc+d$|ef*g.|h?i(j|k))", "EFFGZ"),
        ("(?i)(bc+d$|ef*g.|h?i(j|k))", "IJ"),
        ("(?i)(bc+d$|ef*g.|h?i(j|k))", "EFFG"),
        ("(?i)(bc+d$|ef*g.|h?i(j|k))", "BCDD"),
        ("(?i)(bc+d$|ef*g.|h?i(j|k))", "REFFGZ"),
        ("(?i)((((((((((a))))))))))", "A"),
    ],
)
def test_ignorecase(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("((abc|123)+)!", "!abc123!"),
        ("a+", "xaax"),
        ("(a?)((ab)?)", "ab"),
        ("(a?)((ab)?)(b?)", "ab"),
        ("((a?)((ab)?))(b?)", "ab"),
        ("(a?)(((ab)?)(b?))", "ab"),
        ("(.?)", "x"),
        ("(.?){1}", "x"),
        ("(.?)(.?)", "x"),
        ("(.?){2}", "x"),
        ("(.?.?)", "xxx"),
        ("(.?.?){1}", "xxx"),
        ("(.?.?)(.?.?)", "xxx"),
        ("(.?.?){2}", "xxx"),
        ("(.?.?)(.?.?)(.?.?)", "xxx"),
        ("(.?.?){3}", "xxx"),
        ("a?((ab)?)(b?)", "ab"),
        ("(a?)((ab)?)b?", "ab"),
        ("a?((ab)?)b?", "ab"),
        ("(a*){2}", "xxxxx"),
        ("(ab?)(b?a)", "aba"),
        ("(a|ab)(ba|a)", "aba"),
        ("(a|ab|ba)", "aba"),
        ("(a|ab|ba)(a|ab|ba)", "aba"),
        ("(a|ab|ba)*", "aba"),
        ("(aba|a*b)", "ababa"),
        ("(aba|a*b)(aba|a*b)", "ababa"),
        ("(aba|a*b)(aba|a*b)(aba|a*b)", "ababa"),
        ("(aba|a*b)*", "ababa"),
        ("(aba|ab|a)", "ababa"),
        ("(aba|ab|a)(aba|ab|a)", "ababa"),
        ("(aba|ab|a)(aba|ab|a)(aba|ab|a)", "ababa"),
        ("(aba|ab|a)*", "ababa"),
        ("(a(b)?)", "aba"),
        ("(a(b)?)(a(b)?)", "aba"),
        ("(a(b)?)+", "aba"),
        ("(.*)(.*)", "xx"),
        (".*(.*)", "xx"),
        ("(a.*z|b.*y)", "azbazby"),
        ("(a.*z|b.*y)(a.*z|b.*y)", "azbazby"),
        ("(a.*z|b.*y)*", "azbazby"),
        ("(.|..)(.*)", "ab"),
        ("((..)*(...)*)", "xxx"),
        ("((..)*(...)*)((..)*(...)*)", "xxx"),
        ("(aa(b(b))?)+", "aabbaa"),
        ("(a(b)?)+", "aba"),
        ("([ab]+)([bc]+)([cd]*)", "abcd"),
        ("^(A([^B]*))?(B(.*))?", "Aa"),
        ("^(A([^B]*))?(B(.*))?", "Bb"),
        ("(^){03}", "a"),
        ("($){03}", "a"),
        ("(^){13}", "a"),
        ("($){13}", "a"),
        ("s(()|^)e", "searchme"),
        ("s(^|())e", "searchme"),
        ("s(^|())e", "searchme"),
        ("s()?e", "searchme"),
        ("s(^)?e", "searchme"),
        ("((s)|(e)|(a))*", "searchme"),
        ("(yyy|(x?)){24}", "yyyyyy"),
        ("$()|^()", "ac\\n"),
        ("^()|$()", "ac\\n"),
        ("($)?(.)", "__"),
        ("((a)|(b)){2}", "ab"),
        (".()|((.)?)", "NULL"),
        ("(.|$){2}", "xx"),
        ("(.|$){22}", "xx"),
        ("(.){2}", "xx"),
        ("(a|())(b|())(c|())", "abc"),
        ("ab()c|ab()c()", "abc"),
        ("(b(c)|d(e))*", "bcde"),
        ("(a(b)*)*", "aba"),
    ],
)
def test_groups1(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(.?)*", "x"),  # passes in Golang and Javascript
        ("(.?.?)*", "xxx"),  # passes in Golang and Javascript
        (
            "((s)|(e)|())*",
            "searchme",
        ),  # fix so that empty capturing groups always find a match
        # capturing groups are found
        ("(.|()|())*", "c"),  # empty groups aren't matching
        ("((..)*(...)*)*", "xxx"),  # passes in Golang and Javascript
        ("(a*)*", "a"),  # passes in .NET(C#), Golang, Javascript
        ("a(a*?)(a?)(a??)(a+)(a*)a", "aaaaaa"),  # passes in PCRE2, JS,
    ],
)
def test_ambiguous_cases_groups_pass_on_some_engines(pattern, text):
    _test_case_no_groups(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(a+|b){0,1}?", "ab"),  # passes in .NET(C#), Java8, GoLang, JavaScript
        (
            "((b*)|c(c*))*",
            "cbb",
        ),  # agrees with Javascript, where the first empty string is not found, and the all
    ],
)
@pytest.mark.skip(
    reason="first test passes only in some engines, second test doesn't find first empty string"
)
def test_ambiguous_cases_matches_pass_on_some_engines(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        (r"\)", "()"),
        (r"\}", "}"),
        (r"\]", "]"),  # escaped
        ("$^", "NULL"),
        ("a($)", "aa"),
        ("a*(^a)", "aa"),
        ("(..)*(...)*", "a"),
        ("(..)*(...)*", "abcd"),
        ("(ab|a)(bc|c)", "abc"),
        ("(ab)c|abc", "abc"),
        ("a{0}b", "ab"),
        ("(a*)(b?)(b+)b{3}", "aaabbbbbbb"),
        ("(a*)(b{0,1})(b{1,})b{3}", "aaabbbbbbb"),
        ("((a|a)|a)", "a"),
        ("(a*)(a|aa)", "aaaa"),
        ("a*(a.|aa)", "aaaa"),
        ("a(b)|c(d)|a(e)f", "aef"),
        ("(a|b)?.*", "b"),
        ("(a|b)c|a(b|c)", "ac"),
        ("(a|b)c|a(b|c)", "ab"),
        ("(a|b)*c|(a|ab)*c", "abc"),
        ("(a|b)*c|(a|ab)*c", "xc"),
        ("(.a|.b).*|.*(.a|.b)", "xa"),
        ("a?(ab|ba)ab", "abab"),
        ("((((((((((((((((((((((((((((((x))))))))))))))))))))))))))))))", "x"),
        ("((((((((((((((((((((((((((((((x))))))))))))))))))))))))))))))*", "xx"),
        ("a?(ac{0}b|ba)ab", "abab"),
        ("ab|abab", "abbabab"),
        ("aba|bab|bba", "baaabbbaba"),
        ("aba|bab", "baaabbbaba"),
        ("(aa|aaa)*|(a|aaaaa)", "aa"),
        ("(a.|.a.)*|(a|.a...)", "aa"),
        ("ab|a", "xabc"),
        ("ab|a", "xxabc"),
        ("(Ab|cD)*", "aBcD"),
        (":::1:::0:|:::1:1:0:", ":::0:::1:::1:::0:"),
        (":::1:::0:|:::1:1:1:", ":::0:::1:::1:::0:"),
        ("(a)(b)(c)", "abc"),
        (
            "a?(ab|ba)*",
            "ababababababababababababababababababababababababababababababababababababababababa",
        ),
        ("abaa|abbaa|abbbaa|abbbbaa", "ababbabbbabbbabbbbabbbbaa"),
        ("abaa|abbaa|abbbaa|abbbbaa", "ababbabbbabbbabbbbabaa"),
        ("aaac|aabc|abac|abbc|baac|babc|bbac|bbbc", "baaabbbabac"),
        (
            "aaaa|bbbb|cccc|ddddd|eeeeee|fffffff|gggg|hhhh|iiiii|jjjjj|kkkkk|llll",
            "XaaaXbbbXcccXdddXeeeXfffXgggXhhhXiiiXjjjXkkkXlllXcbaXaaaa",
        ),
        ("a*a*a*a*a*b", "aaaaaaaaab"),
        ("ab+bc", "abbc"),
        ("ab+bc", "abbbbc"),
        ("ab?bc", "abbc"),
        ("ab?bc", "abc"),
        ("ab?c", "abc"),
        ("ab|cd", "abc"),
        ("ab|cd", "abcd"),
        ("a\\(b", "a(b"),
        ("a\\(*b", "ab"),
        ("a\\(*b", "a((b"),
        ("((a))", "abc"),
        ("(a)b(c)", "abc"),
        ("a+b+c", "aabbabc"),
        ("a*", "aaa"),
        ("(a+|b)*", "ab"),
        ("(a+|b)+", "ab"),
        ("(a+|b)?", "ab"),
        ("([abc])*d", "abbbcd"),
        ("([abc])*bcd", "abcd"),
        ("a|b|c|d|e", "e"),
        ("(a|b|c|d|e)f", "ef"),
        ("(ab|cd)e", "abcde"),
        ("(a|b)c*d", "abcd"),
        ("(ab|ab*)bc", "abc"),
        ("a([bc]*)c*", "abc"),
        ("a([bc]*)(c*d)", "abcd"),
        ("a([bc]+)(c*d)", "abcd"),
        ("a([bc]*)(c+d)", "abcd"),
        ("a[bcd]*dcdcde", "adcdcde"),
        ("(ab|a)b*c", "abc"),
        ("((a)(b)c)(d)", "abcd"),
        ("^a(bc+|b[eh])g|.h$", "abh"),
        ("(bc+d$|ef*g.|h?i(j|k))", "effgz"),
        ("(bc+d$|ef*g.|h?i(j|k))", "ij"),
        ("(bc+d$|ef*g.|h?i(j|k))", "reffgz"),
        ("(((((((((a)))))))))", "a"),
        ("(.*)c(.*)", "abcde"),
        ("a(bc)d", "abcd"),
        ("a[\x01-\x03]?c", "a\x02c"),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Qaddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Mo'ammar_Gadhafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Kaddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Qadhafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Gadafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Mu'ammar_Qadafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Moamar_Gaddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Mu'ammar_Qadhdhafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Khaddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Ghaddafy",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Ghadafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Ghaddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muamar_Kaddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Quathafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Muammar_Gheddafi",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Moammar_Khadafy",
        ),
        (
            "M[ou]'?am+[ae]r_.*([AEae]l[-_])?[GKQ]h?[aeu]+([dtz][dhz]?)+af[iy]",
            "Moammar_Qudhafi",
        ),
        ("a+(b|c)*d+", "aabcdd"),
        ("^.+$", "vivi"),
        ("^(.+)$", "vivi"),
        ("^([^!.]+).att.com!(.+)$", "gryphon.att.com!eby"),
        ("^([^!]+!)?([^!]+)$", "bas"),
        ("^([^!]+!)?([^!]+)$", "bar!bas"),
        ("^([^!]+!)?([^!]+)$", "foo!bas"),
        ("^.+!([^!]+!)([^!]+)$", "foo!bar!bas"),
        ("((foo)|(bar))!bas", "bar!bas"),
        ("((foo)|(bar))!bas", "foo!bar!bas"),
        ("((foo)|(bar))!bas", "foo!bas"),
        ("((foo)|bar)!bas", "bar!bas"),
        ("((foo)|bar)!bas", "foo!bar!bas"),
        ("((foo)|bar)!bas", "foo!bas"),
        ("(foo|(bar))!bas", "bar!bas"),
        ("(foo|(bar))!bas", "foo!bar!bas"),
        ("(foo|(bar))!bas", "foo!bas"),
        ("(foo|bar)!bas", "bar!bas"),
        ("(foo|bar)!bas", "foo!bar!bas"),
        ("(foo|bar)!bas", "foo!bas"),
        ("^(([^!]+!)?([^!]+)|.+!([^!]+!)([^!]+))$", "foo!bar!bas"),
        ("^([^!]+!)?([^!]+)$|^.+!([^!]+!)([^!]+)$", "bas"),
        ("^([^!]+!)?([^!]+)$|^.+!([^!]+!)([^!]+)$", "bar!bas"),
        ("^([^!]+!)?([^!]+)$|^.+!([^!]+!)([^!]+)$", "foo!bar!bas"),
        ("^([^!]+!)?([^!]+)$|^.+!([^!]+!)([^!]+)$", "foo!bas"),
        ("^(([^!]+!)?([^!]+)|.+!([^!]+!)([^!]+))$", "bas"),
        ("^(([^!]+!)?([^!]+)|.+!([^!]+!)([^!]+))$", "bar!bas"),
        ("^(([^!]+!)?([^!]+)|.+!([^!]+!)([^!]+))$", "foo!bar!bas"),
        ("^(([^!]+!)?([^!]+)|.+!([^!]+!)([^!]+))$", "foo!bas"),
        (".*(/XXX).*", "/XXX"),
        (".*(\\\\XXX).*", "\\XXX"),
        ("\\\\XXX", "\\XXX"),
        (".*(/000).*", "/000"),
        (".*(\\\\000).*", "\\000"),
        ("\\\\000", "\\000"),
    ],
)
def test_basic3(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(a*)+", "-"),
        ("(?:a|)*", "-"),
        ("(?:a*)*", "-"),
        ("(?:a*)+", "-"),
        ("(?:a*|b)*", "-"),
        ("(?:^)*", "-"),
    ],
)
def test_infinite_loops(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("aa*", "xaxaax"),
        ("(a*)(ab)*(b*)", "abc"),
        ("(a*)(ab)*(b*)", "abc"),
        ("((a*)(ab)*)((b*)(a*))", "aba"),
        ("(...?.?)*", "xxxxxx"),
        ("(a|ab)(bc|c)", "abcabc"),
        ("(aba|a*b)(aba|a*b)", "ababa"),
        ("(a*){2}", "xxxxx"),
        ("(aba|a*b)*", "ababa"),
        ("(a(b)?)+", "aba"),
        (".*(.*)", "ab"),
        ("(a?)((ab)?)(b?)a?(ab)?b?", "abab"),
        ("(a?)((ab)?)(b?)a?(ab)?b?", "abab"),
    ],
)
def test_class(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(a|ab)(c|bcd)", "abcd"),
        ("(a|ab)(bcd|c)", "abcd"),
        ("(ab|a)(c|bcd)", "abcd"),
        ("(ab|a)(bcd|c)", "abcd"),
        ("((a|ab)(c|bcd))(d*)", "abcd"),
        ("((a|ab)(bcd|c))(d*)", "abcd"),
        ("((ab|a)(c|bcd))(d*)", "abcd"),
        ("((ab|a)(bcd|c))(d*)", "abcd"),
        ("(a|ab)((c|bcd)(d*))", "abcd"),
        ("(a|ab)((bcd|c)(d*))", "abcd"),
        ("(ab|a)((c|bcd)(d*))", "abcd"),
        ("(ab|a)((bcd|c)(d*))", "abcd"),
        ("(a*)(b|abc)", "abc"),
        ("(a*)(abc|b)", "abc"),
        ("((a*)(b|abc))(c*)", "abc"),
        ("((a*)(abc|b))(c*)", "abc"),
        ("(a*)((b|abc)(c*))", "abc"),
        ("(a*)((abc|b)(c*))", "abc"),
        ("(a*)(b|abc)", "abc"),
        ("(a*)(abc|b)", "abc"),
        ("((a*)(b|abc))(c*)", "abc"),
        ("((a*)(abc|b))(c*)", "abc"),
        ("(a*)((b|abc)(c*))", "abc"),
        ("(a*)((abc|b)(c*))", "abc"),
        ("(a|ab)", "ab"),
        ("(ab|a)", "ab"),
        ("(a|ab)(b*)", "ab"),
        ("(ab|a)(b*)", "ab"),
    ],
)
def test_forced_assoc(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(a|ab)(c|bcd)(d*)", "abcd"),
        ("(a|ab)(bcd|c)(d*)", "abcd"),
        ("(ab|a)(c|bcd)(d*)", "abcd"),
        ("(ab|a)(bcd|c)(d*)", "abcd"),
        ("(a*)(b|abc)(c*)", "abc"),
        ("(a*)(abc|b)(c*)", "abc"),
        ("(a*)(b|abc)(c*)", "abc"),
        ("(a*)(abc|b)(c*)", "abc"),
        ("(a|ab)(c|bcd)(d|.*)", "abcd"),
        ("(a|ab)(bcd|c)(d|.*)", "abcd"),
        ("(ab|a)(c|bcd)(d|.*)", "abcd"),
        ("(ab|a)(bcd|c)(d|.*)", "abcd"),
    ],
)
def test_left_assoc(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(a*)*", "a"),
        ("(a*)*", "x"),
        ("(a*)*", "aaaaaa"),
        ("(a*)*", "aaaaaax"),
        ("(a*)+", "a"),
        ("(a*)+", "x"),
        ("(a*)+", "aaaaaa"),
        ("(a*)+", "aaaaaax"),
        ("(a+)*", "a"),
        ("(a+)*", "x"),
        ("(a+)*", "aaaaaa"),
        ("(a+)*", "aaaaaax"),
        ("(a+)+", "a"),
        ("(a+)+", "x"),
        ("(a+)+", "aaaaaa"),
        ("(a+)+", "aaaaaax"),
        ("([a]*)*", "a"),
        ("([a]*)*", "x"),
        ("([a]*)*", "aaaaaa"),
        ("([a]*)*", "aaaaaax"),
        ("([a]*)+", "a"),
        ("([a]*)+", "x"),
        ("([a]*)+", "aaaaaa"),
        ("([a]*)+", "aaaaaax"),
        ("([^b]*)*", "a"),
        ("([^b]*)*", "b"),
        ("([^b]*)*", "aaaaaa"),
        ("([^b]*)*", "aaaaaab"),
        ("([ab]*)*", "a"),
        ("([ab]*)*", "aaaaaa"),
        ("([ab]*)*", "ababab"),
        ("([ab]*)*", "bababa"),
        ("([ab]*)*", "b"),
        ("([ab]*)*", "bbbbbb"),
        ("([ab]*)*", "aaaabcde"),
        ("([^a]*)*", "b"),
        ("([^a]*)*", "bbbbbb"),
        ("([^a]*)*", "aaaaaa"),
        ("([^ab]*)*", "ccccxx"),
        ("([^ab]*)*", "ababab"),
        ("((z)+|a)*", "zabcde"),
        ("(a)", "aaa"),
        ("(a*)*(x)", "x"),
        ("(a*)*(x)", "ax"),
        ("(a*)*(x)", "axa"),
        ("(a*)+(x)", "x"),
        ("(a*)+(x)", "ax"),
        ("(a*)+(x)", "axa"),
        ("(a*){2}(x)", "x"),
        ("(a*){2}(x)", "ax"),
        ("(a*){2}(x)", "axa"),
    ],
)
def test_null_sub3(pattern, text):
    _test_case_no_groups(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(()|.)(b)", "ab"),
        ("(()|.)(b)", "ab"),
        ("(()|[ab])(b)", "ab"),
        ("([ab]|())(b)", "ab"),
        ("(.?)(b)", "ab"),
    ],
)
def test_osx_bsd_critical(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(()|[ab])+b", "aaab"),
        ("(.|())(b)", "ab"),
        ("([ab]|())+b", "aaab"),
    ],
)
def test_osx_bsd_critical_no_groups(pattern, text):
    _test_case_no_groups(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("((..)|(.))", "NULL"),
        ("((..)|(.))((..)|(.))", "NULL"),
        ("((..)|(.))((..)|(.))((..)|(.))", "NULL"),
        ("((..)|(.)){1}", "NULL"),
        ("((..)|(.)){2}", "NULL"),
        ("((..)|(.)){3}", "NULL"),
        ("((..)|(.))*", "NULL"),
        ("((..)|(.))", "a"),
        ("((..)|(.))((..)|(.))", "a"),
        ("((..)|(.))((..)|(.))((..)|(.))", "a"),
        ("((..)|(.)){1}", "a"),
        ("((..)|(.)){2}", "a"),
        ("((..)|(.)){3}", "a"),
        ("((..)|(.))*", "a"),
        ("((..)|(.))", "aa"),
        ("((..)|(.))((..)|(.))", "aa"),
        ("((..)|(.))((..)|(.))((..)|(.))", "aa"),
        ("((..)|(.)){1}", "aa"),
        ("((..)|(.)){2}", "aa"),
        ("((..)|(.)){3}", "aa"),
        ("((..)|(.))*", "aa"),
        ("((..)|(.))", "aaa"),
        ("((..)|(.))((..)|(.))", "aaa"),
        ("((..)|(.))((..)|(.))((..)|(.))", "aaa"),
        ("((..)|(.)){1}", "aaa"),
        ("((..)|(.)){2}", "aaa"),
        ("((..)|(.)){3}", "aaa"),
        ("((..)|(.))*", "aaa"),
        ("((..)|(.))", "aaaa"),
        ("((..)|(.))((..)|(.))", "aaaa"),
        ("((..)|(.))((..)|(.))((..)|(.))", "aaaa"),
        ("((..)|(.)){1}", "aaaa"),
        ("((..)|(.)){2}", "aaaa"),
        ("((..)|(.)){3}", "aaaa"),
        ("((..)|(.))*", "aaaa"),
        ("((..)|(.))", "aaaaa"),
        ("((..)|(.))((..)|(.))", "aaaaa"),
        ("((..)|(.))((..)|(.))((..)|(.))", "aaaaa"),
        ("((..)|(.)){1}", "aaaaa"),
        ("((..)|(.)){2}", "aaaaa"),
        ("((..)|(.)){3}", "aaaaa"),
        ("((..)|(.))*", "aaaaa"),
        ("((..)|(.))", "aaaaaa"),
        ("((..)|(.))((..)|(.))", "aaaaaa"),
        ("((..)|(.))((..)|(.))((..)|(.))", "aaaaaa"),
        ("((..)|(.)){1}", "aaaaaa"),
        ("((..)|(.)){2}", "aaaaaa"),
        ("((..)|(.)){3}", "aaaaaa"),
        ("((..)|(.))*", "aaaaaa"),
        ("X(.?){8,}Y", "X1234567Y"),
        ("X(.?){0,8}Y", "X1234567Y"),
        ("X(.?){1,8}Y", "X1234567Y"),
        ("X(.?){2,8}Y", "X1234567Y"),
        ("X(.?){3,8}Y", "X1234567Y"),
        ("X(.?){4,8}Y", "X1234567Y"),
        ("X(.?){5,8}Y", "X1234567Y"),
        ("X(.?){6,8}Y", "X1234567Y"),
        ("X(.?){7,8}Y", "X1234567Y"),
        ("X(.?){8,8}Y", "X1234567Y"),
        ("(a|ab|c|bcd){0,}(d*)", "ababcd"),
        ("(a|ab|c|bcd){1,}(d*)", "ababcd"),
        ("(a|ab|c|bcd){2,}(d*)", "ababcd"),
        ("(a|ab|c|bcd){3,}(d*)", "ababcd"),
        ("(a|ab|c|bcd){4,}(d*)", "ababcd"),
        ("(a|ab|c|bcd){0,10}(d*)", "ababcd"),
        ("(a|ab|c|bcd){1,10}(d*)", "ababcd"),
        ("(a|ab|c|bcd){2,10}(d*)", "ababcd"),
        ("(a|ab|c|bcd){3,10}(d*)", "ababcd"),
        ("(a|ab|c|bcd){4,10}(d*)", "ababcd"),
        ("(a|ab|c|bcd)*(d*)", "ababcd"),
        ("(a|ab|c|bcd)+(d*)", "ababcd"),
    ],
)
def test_repetition2(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("X(.?){0,}Y", "X1234567Y"),
        ("X(.?){1,}Y", "X1234567Y"),
        ("X(.?){2,}Y", "X1234567Y"),
        ("X(.?){3,}Y", "X1234567Y"),
        ("X(.?){4,}Y", "X1234567Y"),
        ("X(.?){5,}Y", "X1234567Y"),
        ("X(.?){6,}Y", "X1234567Y"),
        ("X(.?){7,}Y", "X1234567Y"),
    ],
)
def test_repetition2_no_groups(pattern, text):
    # Golang, Javascript capture 7, Python doesn't
    _test_case_no_groups(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(a|ab)(c|bcd)(d*)", "abcd"),
        ("(a|ab)(bcd|c)(d*)", "abcd"),
        ("(ab|a)(c|bcd)(d*)", "abcd"),
        ("(ab|a)(bcd|c)(d*)", "abcd"),
        ("(a*)(b|abc)(c*)", "abc"),
        ("(a*)(abc|b)(c*)", "abc"),
        ("(a*)(b|abc)(c*)", "abc"),
        ("(a*)(abc|b)(c*)", "abc"),
        ("(a|ab)(c|bcd)(d|.*)", "abcd"),
        ("(a|ab)(bcd|c)(d|.*)", "abcd"),
        ("(ab|a)(c|bcd)(d|.*)", "abcd"),
        ("(ab|a)(bcd|c)(d|.*)", "abcd"),
    ],
)
def test_right_assoc(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(?m)^abc", "abcdef"),
        ("(?m)^abc", "aabcdef"),
        ("(?m)^[ay]*[bx]+c", "abcdef"),
        ("(?m)^[ay]*[bx]+c", "aabcdef"),
        ("(?m)def$", "abcdef"),
        ("(?m)def$", "abcdeff"),
        ("(?m)d[ex][fy]$", "abcdef"),
        ("(?m)d[ex][fy]$", "abcdeff"),
        ("(?m)[dz][ex][fy]$", "abcdef"),
        ("(?m)[dz][ex][fy]$", "abcdeff"),
        ("(?m)^b$", "a\nb\nc\n"),
        ("(foo|bar|[A-Z])$", "foo\n"),
        ("(foo|bar|[A-Z])$", "foo\nbar"),  # should find no match
        ("^(foo|bar|[A-Z])$", "foo\nbarfoo"),  # no matches
        ("(?m)^(foo|bar|[A-Z])$", "foo\nbar"),  # should find matches
        ("(?m)^b", "a\nb\n"),
        ("(?m)^(b)", "a\nb\n"),
        ("(?m)\n(^b)", "a\nb\n"),
        ("foo.$", "foo1\nfoo2\n"),
        ("(?m)foo.$", "foo1\nfoo2\n"),
        ("$", "foo\n"),
    ],
)
def test_multiline(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("\\A(foo|bar|[A-Z])$", "foo\n"),
        ("\\A(foo|bar|[A-Z])$", "foo\nbar"),  # should find no match
        ("\\A^(foo|bar|[A-Z])$", "foo\nbarfoo"),  # no matches
        ("(?m)\\A^(foo|bar|[A-Z])$", "foo\nbar"),  # should find matches
        ("(?m)\\A^b", "a\nb\n"),
        ("(?m)\\A^(b)", "a\nb\n"),
        ("(?m)\\A\n(^b)", "a\nb\n"),
        ("\\A^b", "a\nb\n"),
        ("\\A^(b)", "a\nb\n"),
        ("\\A\n(^b)", "a\nb\n"),
    ],
)
def test_start_of_string_absolute_anchor(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text, expected",
    [
        ("StackOverflow\\Z", "StackOverflow\n", ["StackOverflow"]),
        ("StackOverflow\\z", "StackOverflow\n", []),
    ],
)
def test_end_of_string_absolute_anchors(pattern, text, expected):
    actual = RegexPikeVM(pattern).findall(text)
    assert actual == expected


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("(?:(a*|b))*", "-"),
        ("(a|)*", "-"),
        ("((a*|b))*", "-"),
        ("(^)*", "-"),
        ("(a*|b)*", "-"),
        ("(a*)*", "-"),
        ("($)|()", "xxx"),
        ("((s^)|(s)|(^)|($)|(^.))*", "searchme"),
    ],
)
def test_groups_failing_in_vm_passing_in_nfa(pattern, text):
    _test_case_no_groups(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        (r"^([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6})*$", email)
        for email in [
            "email@example.com",
            "firstname.lastname@example.com",
            "email@subdomain.example.com",
            "firstname+lastname@example.com",  # this is actually valid
            "email@123.123.123.123",  # valid
            "email@[123.123.123.123]",  # valid
            'email"@example.com',  # valid
            "1234567890@example.com",
            "email@example-one.com",
            "_______@example.com",
            "email@example.name",
            "email@example.museum",
            "email@example.co.jp",
            "firstname-lastname@example.com",
        ]
    ],
)
def test_common_email(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        (
            r"^https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#()?&\/=]*)$",
            url,
        )
        for url in [
            "http://foo.com/blah_blah",
            "http://foo.com/blah_blah/",
            "http://foo.com/blah_blah_(wikipedia)",
            "http://www.example.com/wpstyle/?p=364",
            "https://www.example.com/foo/?bar=baz&inga=42&quux",
            "http://userid:password@example.com:8080",
            "http://foo.com/blah_(wikipedia)#cite-1",
            "www.google.com",
            "http://../",
            "http:// shouldfail.com",
            "http://224.1.1.1",
            "http://142.42.1.1:8080/",
            "ftp://foo.bar/baz",
            "http://1337.net",
            "http://foo.bar/?q=Test%20URL-encoded%20stuff",
            "http://code.google.com/events/#&product=browser",
            "http://-error-.invalid/",
            "http://3628126748",
            "http://उदाहरण.परीक्षा",
        ]
    ],
)
def test_urls(pattern, text):
    _test_case_no_groups(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [
        (
            r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$",
            ipv4_address,
        )
        for ipv4_address in [
            "0.0.0.0",
            "9.255.255.255",
            "11.0.0.0",
            "126.255.255.255",
            "129.0.0.0",
            "169.253.255.255",
            "169.255.0.0",
            "172.15.255.255",
            "172.32.0.0",
            "256.0.0.0",  # not a valid address
            "191.0.1.255",
            "192.88.98.255",
            "192.88.100.0",
            "192.167.255.255",
            "192.169.0.0",
            "198.17.255.255",
            "223.255.255.255",
        ]
    ],
)
def test_ipv4_addresses(pattern, text):
    _test_case(pattern, text)


@pytest.mark.parametrize(
    "pattern, text",
    [("((?:[a-zA-Z0-9]{2}[:-]){5}[a-zA-Z0-9]{2})", "00:0a:95:9d:68:16")],
)
def test_mac_address(pattern, text):
    _test_case(pattern, text)
