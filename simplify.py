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


def simplify(regexp: str) -> str:
    return simplify_redundant_quantifiers(regexp)
