from string import ascii_lowercase, ascii_uppercase, digits
from symbol import Character, MetaSequence, OneOf, Operator, Symbol, Caret

from symbol import AllOps, ESCAPED


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
        special_char not in regexp for special_char in ("|", "?", "+", "*", "(", ")")
    )
    negated = False
    if regexp[1] == Caret:
        negated = True
    original_regexp = regexp
    regexp = regexp[:1] + regexp[2:]
    if "-" in regexp:
        simplified = simplify_ranges(regexp)
    else:
        simplified = regexp[1:-1]
    return OneOf(set(simplified), original_regexp, negated=negated)


def simplify_ranges(regexp: str):
    assert regexp[0] == "[" and regexp[-1] == "]"
    assert regexp.count("[") == 1 and regexp.count("]") == 1
    assert len(regexp) > 1

    len_regexp = len(regexp)
    i = 1
    subs: list[str] = []
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


def parse_repetition_range(symbols: list[Symbol], regexp: str) -> list[Symbol]:
    print(symbols, regexp)
    pass


def to_symbols(regexp: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    index = 0
    while index < len(regexp):
        char = regexp[index]
        if char in AllOps:
            symbols.append(Operator(char))
        elif char == "\\":
            next_char = regexp[index + 1]  # index out of bounds?
            if next_char in ESCAPED:
                symbols.append(Character(next_char))
            else:
                symbols.append(MetaSequence(next_char))
                index += 1
        elif char == "[":
            start = index
            index += 1
            while index < len(regexp):
                if regexp[index] == "[":
                    raise ValueError(
                        f"found another opening bracket before the one at {start} was closed"
                    )
                if regexp[index] == "]":
                    symbols.append(parse_character_class(regexp[start : index + 1]))
                    break
                index += 1
            if index >= len(regexp):
                raise ValueError(f"bracket starting at {start}")
        elif char == "{":
            start = index
            index += 1
            while index < len(regexp) and regexp[index] != "}":
                index += 1
            if index >= len(regexp):
                raise ValueError(f"unclosed brace starting at {start}")
            symbols.extend(parse_repetition_range(symbols, regexp[start : index + 1]))

        else:
            symbols.append(Character(char))
        index += 1

    return symbols


def simplify(regexp: str) -> list[Symbol]:
    return to_symbols(simplify_redundant_quantifiers(regexp))
