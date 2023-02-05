import json
from pprint import pprint

from src.core import Transition
from src.match import Regexp
from src.parser import RegexNode


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Transition):
            return tuple(obj)
        if isinstance(obj, RegexNode):
            return obj.string()

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    regex = r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
    pattern = Regexp(regex)
    pprint(json.dumps(pattern, cls=ComplexEncoder))
    pprint(pattern)
