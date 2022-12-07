def substitute_character_classes(regexp: str):
    return (
        regexp.replace("\\d", "[0-9]")
        .replace("\\w", "[A-z0-9_]")
        .replace("\\s", "[ \t\n\r\v\f]")
    )


def apply_replacements(regexp: str):
    return (
        regexp.replace("*?", "*")
        .replace("+*", "*")
        .replace("+?", "*")
        .replace("?*", "*")
        .replace("?+", "*")
        .replace("*+", "*")
        .replace("++", "+")
        .replace("??", "?")
        .replace("()", "")
        .replace("|)", ")|")
    )


def simplify_redundant_quantifiers(regexp: str):
    reduced = apply_replacements(regexp)

    while reduced != regexp:
        regexp = reduced
        reduced = apply_replacements(regexp)

    return reduced


def simplify(regexp: str) -> str:
    return simplify_redundant_quantifiers(regexp)
