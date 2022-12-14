import re

import pytest

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
        expected = [m.group(0) for m in re.finditer(pattern, text)]
        actual = [m.substr for m in RegexMatcher(pattern, text)]
        try:
            assert expected == actual, (i, pattern, text)
        except AssertionError as e:
            print()
            raise e


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
        # (r"(fo|foo)", "fo"),
        # (r"(foo|fo)", "foo"),
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
    ]

    for pattern, text in cases:
        with pytest.raises(re.error):
            _ = [m.group(0) for m in re.finditer(pattern, text) if m.group(0) != ""]
        with pytest.raises(ValueError):
            _ = [m.substr for m in RegexMatcher(pattern, text) if m.substr != ""]


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
    ]

    _test_cases_suite(cases)


def test_unicode_simple():
    cases = [("🇺🇸+", "🇺🇸🇺🇸🇺🇸🇺🇸🇺🇸")]

    _test_cases_suite(cases)
