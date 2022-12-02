from string import ascii_uppercase, ascii_lowercase, digits

from utils import ALL_OPERATORS, LEFT_PAREN, UNION, RIGHT_PAREN


def simplify_kleene_plus(regexp: str) -> str:
    """A+ -> AA*"""
    len_regexp = len(regexp)
    i = 0
    while i < len_regexp:
        if regexp[i] == "+":
            # two cases can arise:
            # 1. + is after a symbol, in which case + affects only that symbol
            # 2. + is after a ')', in which case + affects a whole expression inside '(' and ')'

            if (symbol := regexp[i - 1]) != ")":
                assert regexp[i - 1] not in ALL_OPERATORS and regexp[i - 1] not in (
                    "(",
                    ")",
                )
                sub_exp = symbol + symbol + "*"
                regexp = regexp[: i - 1] + sub_exp + regexp[i + 1 :]
                i += 2  # we added two extra characters
            else:
                # first find lower bound
                bracket_counter = 0
                for j in reversed(range(i)):
                    if j != i - 1 and regexp[j] == ")":
                        bracket_counter += 1
                    # stop
                    if regexp[j] == "(":
                        if bracket_counter != 0:
                            bracket_counter -= 1
                        else:
                            symbol_sequence_with_brackets = regexp[j:i]
                            sub_exp = (
                                symbol_sequence_with_brackets
                                + symbol_sequence_with_brackets
                                + "*"
                            )

                            left = regexp[:j]
                            right = regexp[i + 1 :]
                            regexp = left + sub_exp + right
                            i += len(symbol_sequence_with_brackets)
            len_regexp = len(regexp)
        i += 1
    return regexp


def simplify_lua(regexp: str) -> str:
    """A? represents A|ε"""
    len_regexp = len(regexp)
    i = 0
    while i < len_regexp:
        if regexp[i] == "?":
            # two cases can arise:
            # 1. ? is after a symbol, in which case ? affects only that symbol
            # 2. ? is after a `)`, in which case ? affects a whole expression inside '(' and ')'
            if (symbol := regexp[i - 1]) != ")":
                sub_exp = "(" + symbol + "|ε)"
                regexp = regexp[: i - 1] + sub_exp + regexp[i + 1 :]
                i += 3  # we replaced 1 character with 4, so we advance i by 3
            else:
                assert regexp[i - 1] == ")"
                for j in reversed(range(i)):
                    if regexp[j] == "(":
                        symbol_sequence_with_brackets = regexp[j:i]
                        sub_exp = "(" + symbol_sequence_with_brackets + "|ε)"
                        regexp = regexp[:j] + sub_exp + regexp[i + 1 :]
                        i += 3
                        break
            len_regexp = len(regexp)
        i += 1
    return regexp


def handle_alpha(char_start, chart_end):
    if char_start.isalpha() and chart_end.isalpha():
        letters_sorted = ascii_uppercase + ascii_lowercase
        if char_start > chart_end:
            raise ValueError(f"illegal character range: [{char_start}-{chart_end}]")
        else:
            return letters_sorted[
                letters_sorted.index(char_start) : letters_sorted.index(chart_end) + 1
            ]


def handle_digits_case(char_start, char_end):
    if char_start.isdigit() and char_end.isdigit():
        if char_start > char_end:
            raise ValueError(f"illegal character range: [{char_start}-{char_end}]")
        else:
            return digits[digits.index(char_start) : digits.index(char_end) + 1]


def handle_range(char_start, char_end):
    if not char_start.isalnum() or not char_end.isalnum():
        raise ValueError(f"Illegal character range: [{char_start}-{char_end}]")
    # same letter
    if char_start == char_end:
        return char_start

    return handle_alpha(char_start, char_end) or handle_digits_case(
        char_start, char_end
    )


def parse_character_class(regexp: str):
    if len(regexp) < 2:
        raise ValueError(f"invalid character class{regexp}")
    assert regexp[0] == "[" and regexp[-1] == "]"
    assert all(
        special_char not in regexp for special_char in ALL_OPERATORS + ("(", ")")
    )
    if "-" in regexp:
        regexp = simplify_ranges(regexp)
    else:
        regexp = regexp[1:-1]
    return LEFT_PAREN + UNION.join(regexp) + RIGHT_PAREN


def simplify_ranges(regexp: str):
    assert regexp[0] == "[" and regexp[-1] == "]"
    assert regexp.count("[") == 1 and regexp.count("]") == 1
    assert len(regexp) > 1

    len_regexp = len(regexp)
    i = 1
    subs = []
    while i < len_regexp - 1:
        if regexp[i] == "-":
            sub_regexp = handle_range(regexp[i - 1], regexp[i + 1])
            subs.pop()
            subs.append(sub_regexp)
            i += 2
        else:
            subs.append(regexp[i])
            i += 1
    return "".join(subs)


def substitute_character_classes(regexp: str):
    return (
        regexp.replace("\\d", "[0-9]")
        .replace("\\w", "[A-z0-9_]")
        .replace("\\s", "[ \t\n\r\v\f]")
    )


def simplify_redundant_quantifiers(regexp: str):
    reduced = (
        regexp.replace("**", "*")
        .replace("*?", "*")
        .replace("+*", "*")
        .replace("+?", "*")
        .replace("?*", "*")
        .replace("?+", "*")
        .replace("*+", "*")
        .replace("++", "+")
        .replace("??", "?")
    )
    while reduced != regexp:
        regexp = reduced
        reduced = (
            regexp.replace("**", "*")
            .replace("*?", "*")
            .replace("+*", "*")
            .replace("+?", "*")
            .replace("?*", "*")
            .replace("?+", "*")
            .replace("*+", "*")
            .replace("++", "+")
            .replace("??", "?")
        )

    return reduced


def simplify_character_classes(regexp: str):
    regexp = substitute_character_classes(regexp)
    subs = []
    s = None
    for i in range(len(regexp)):
        if regexp[i] == "]":
            if s is not None:
                sub_regexp = parse_character_class(regexp[s : i + 1])
                subs.append(sub_regexp)
                s = None
            else:
                raise ValueError(
                    f"unescaped closing bracket found at regexp[{i}] before any opening bracket"
                )
        elif regexp[i] == "[":
            if s is not None:
                raise ValueError(
                    f"found another opening bracket before the one at {s} was closed"
                )
            s = i
        elif s is None:
            subs.append(regexp[i])
    if s is not None:
        raise ValueError(
            f"could not find closing square bracket to the one opened at regexp[{s}]"
        )
    return "".join(subs)


def simplify(regexp: str) -> str:
    return simplify_redundant_quantifiers(simplify_character_classes(regexp))
