import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.state import MyUsage
from pydantic_ai.usage import Usage
from _utils import update_usage


def test_update_usage_sums_fields():
    current = MyUsage(requests=1, request_tokens=2, response_tokens=3, total_tokens=4)
    new_usage = Usage(requests=4, request_tokens=5, response_tokens=6, total_tokens=7)

    result = update_usage(current, new_usage)

    assert result is current
    assert result.requests == 5
    assert result.request_tokens == 7
    assert result.response_tokens == 9
    assert result.total_tokens == 11
