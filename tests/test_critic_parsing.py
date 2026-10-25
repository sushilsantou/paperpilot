"""
Pure tests of how the critic reads the judge's verdict. A verdict it cannot
read is treated as a failed check (see test_agent_graph_e2e.py), so the parser
has to be tolerant of harmless formatting and strict about the fields the
routing depends on.
"""
import pytest

from src.agents.critic_agent import _parse_judge_json

VERDICT = '{"faithful": true, "issues": "", "sufficient_context": true}'


def test_judge_json_is_parsed_through_fences_and_surrounding_prose():
    expected = {"faithful": True, "issues": "", "sufficient_context": True}

    assert _parse_judge_json(VERDICT) == expected
    assert _parse_judge_json(f"```json\n{VERDICT}\n```") == expected
    assert _parse_judge_json(f"Here is my verdict:\n{VERDICT}\nHope that helps.") == expected


def test_judge_verdict_without_usable_fields_is_rejected():
    """A string "true" is truthy, so accepting it would route on a field the
    judge never actually set to a boolean."""
    malformed = [
        "I think this looks fine, honestly",               # no object at all
        "[true]",                                           # JSON, but not an object
        '{"issues": "", "sufficient_context": true}',       # faithful missing
        '{"faithful": "true", "sufficient_context": true}', # faithful not a bool
        '{"faithful": true, "sufficient_context": "no"}',   # sufficient_context not a bool
    ]
    for raw in malformed:
        with pytest.raises(ValueError):
            _parse_judge_json(raw)
