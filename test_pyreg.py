import logging
import re
from random import randint, random, seed

import pytest

from core import InvalidCharacterRange
from match import Regexp

logging.basicConfig(filename="test.log", level=logging.NOTSET, encoding="utf-8")


# acquired from re2: https://github.com/google/re2/blob/main/re2/testing/search_test.cc


def _test_cases_suite(cases: list[tuple[str, str]]):
    for i, (pattern, text) in enumerate(cases):
        expected = [m.group(0) for m in re.finditer(pattern, text)]
        actual = [m.group(0) for m in Regexp(pattern).finditer(text)]
        assert expected == actual, (i, pattern, text)

        expected_groups = [m.groups() for m in re.finditer(pattern, text)]
        actual_groups = [m.all_groups() for m in Regexp(pattern).finditer(text)]
        print((i, pattern, text, expected_groups, actual_groups))
        for group, all_groups in zip(expected_groups, actual_groups):
            assert group in all_groups, (i, pattern, text)
            logging.info(
                f"{i: 0f} pattern = {pattern!r}, text = {text!r}, groups={group}, all_groups={all_groups}"
            )


def test_repetition():
    cases = [
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
    ]
    _test_cases_suite(cases)


def test_interesting_cases():
    cases = [
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
        ("", ""),  # empty string
    ]

    _test_cases_suite(cases)


def test_python_benchmark():
    cases = [
        # test common prefix
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
    ]

    _test_cases_suite(cases)


[SUCCEED, FAIL, SYNTAX_ERROR] = range(3)


def test_raises_exception():
    cases = [
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
    ]

    for pattern, text in cases:
        with pytest.raises(re.error):
            _ = [m.group(0) for m in re.finditer(pattern, text) if m.group(0) != ""]
        with pytest.raises((ValueError, InvalidCharacterRange)):
            _ = [m.substr for m in Regexp(pattern).finditer(text) if m.substr != ""]


def test_failures():
    cases = [
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
    ]

    _test_cases_suite(cases)


def test_more_python_re_implementation_cases():
    cases = [
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
    ]

    _test_cases_suite(cases)


def test_greedy_vs_lazy():
    cases = [
        ("a.+?c", "abcabc"),
        ("a.*?c", "abcabc"),
        ("a.{0,5}?c", "abcabc"),
        ("a{2,3}?", "aaaaa"),
        # ("(a+|b){0,1}?", "ab"),
    ]

    _test_cases_suite(cases)


seed(10)


def test_hex():
    pattern = r"0[xX](_?[0-9a-fA-F])+"
    cases = [(pattern, hex(randint(0, 100000))) for _ in range(20)]

    _test_cases_suite(cases)


def test_dec():
    pattern = r"(0(_?0)*|[1-9](_?[0-9])*)"
    cases = [(pattern, str(randint(0, 100000))) for _ in range(20)]

    _test_cases_suite(cases)


def test_point_float():
    def group(*choices):
        return "(" + "|".join(choices) + ")"

    def maybe(*choices):
        return group(*choices) + "?"

    exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
    pointfloat = group(
        r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
    ) + maybe(exponent)

    cases = (
        [(pointfloat, str(random())) for _ in range(20)]
        + [(pointfloat, f"{random():.2E}") for _ in range(20)]
        + [(pointfloat, f"{-random():.2E}") for _ in range(20)]
        + [(pointfloat, f"{-random()}") for _ in range(20)]
        + [(pointfloat, f"{10 + random()}") for _ in range(20)]
    )

    _test_cases_suite(cases)


def test_unicode_simple():
    cases = [("🇺🇸+", "🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸")]

    _test_cases_suite(cases)


def test_ignorecase():
    cases = [
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
        # ("(?i)(a+|b){0,1}?", "AB"),
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
    ]

    _test_cases_suite(cases)


@pytest.mark.skip
def test_groups1():
    cases = [
        ("a+", "xaax"),
        ("(a?)((ab)?)", "ab"),
        ("(a?)((ab)?)(b?)", "ab"),
        ("((a?)((ab)?))(b?)", "ab"),
        ("(a?)(((ab)?)(b?))", "ab"),
        ("(.?)", "x"),
        ("(.?){1}", "x"),
        ("(.?)(.?)", "x"),
        ("(.?){2}", "x"),
        ("(.?)*", "x"),
        ("(.?.?)", "xxx"),
        ("(.?.?){1}", "xxx"),
        ("(.?.?)(.?.?)", "xxx"),
        ("(.?.?){2}", "xxx"),
        ("(.?.?)(.?.?)(.?.?)", "xxx"),
        ("(.?.?){3}", "xxx"),
        ("(.?.?)*", "xxx"),
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
        ("((..)*(...)*)*", "xxx"),
        ("(aa(b(b))?)+", "aabbaa"),
        ("(a(b)?)+", "aba"),
        ("([ab]+)([bc]+)([cd]*)", "abcd"),
        ("^(A([^B]*))?(B(.*))?", "Aa"),
        ("^(A([^B]*))?(B(.*))?", "Bb"),
        ("(^){03}", "a"),
        ("($){03}", "a"),
        ("(^){13}", "a"),
        ("($){13}", "a"),
        ("((s^)|(s)|(^)|($)|(^.))*", "searchme"),
        ("s(()|^)e", "searchme"),
        ("s(^|())e", "searchme"),
        ("s(^|())e", "searchme"),
        ("s()?e", "searchme"),
        ("s(^)?e", "searchme"),
        ("((s)|(e)|(a))*", "searchme"),
        ("((s)|(e)|())*", "searchme"),
        ("((b*)|c(c*))*", "cbb"),
        ("(yyy|(x?)){24}", "yyyyyy"),
        ("($)|()", "xxx"),
        ("$()|^()", "ac\\n"),
        ("^()|$()", "ac\\n"),
        ("($)?(.)", "__"),
        ("(.|()|())*", "c"),
        ("((a)|(b)){2}", "ab"),
        (".()|((.)?)", "NULL"),
        ("(.|$){2}", "xx"),
        ("(.|$){22}", "xx"),
        ("(.){2}", "xx"),
        ("(a|())(b|())(c|())", "abc"),
        ("ab()c|ab()c()", "abc"),
        ("(b(c)|d(e))*", "bcde"),
        ("(a(b)*)*", "aba"),
    ]

    _test_cases_suite(cases)
