

from __future__ import annotations

import json
from datasets import Dataset

from verifiers.envs.bfcl_v4_env import _normalize_answer, BFCLV4SingleTurnEnv


def test_v4_normalization_basic_punctuation():
    """Test basic punctuation removal as specified by BFCL v4."""
    # Test the exact punctuation set mentioned in BFCL v4: , . / - _ * ^ ( ) and quotes
    test_cases = [
        ("Hello, World.", "hello world"),  # Comma and period
        ("path/to/file", "pathtofile"),    # Forward slash
        ("key-value", "keyvalue"),        # Dash
        ("variable_name", "variablename"), # Underscore
        ("2*2=4", "22=4"),              # Asterisk, = sign preserved
        ("raise^to_power", "raisetopower"), # Caret
        ("function(arg)", "functionarg"),  # Parentheses
        ('quote"test', 'quotetest'),      # Double quote
        ("apostrophe'test", "apostrophetest"),  # Single quote
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_lowercase():
    """Test that normalization converts to lowercase."""
    test_cases = [
        ("HELLO WORLD", "hello world"),
        ("CamelCase", "camelcase"),
        ("MiXeD CaSe", "mixed case"),
        ("UPPER", "upper"),
        ("lower", "lower"),
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_no_diacritics():
    """Test that diacritics are NOT removed (BFCL-exact behavior)."""
    # BFCL v4 spec does NOT include diacritic removal
    test_cases = [
        ("résumé", "résumé"),        # Should keep diacritics
        ("café", "café"),            # Should keep diacritics
        ("naïve", "naïve"),          # Should keep diacritics
        ("Eiffel Tower", "eiffel tower"),  # Regular text should be lowercased
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_no_unicode_nfc():
    """Test that Unicode NFC normalization is NOT applied (BFCL-exact behavior)."""
    # BFCL v4 spec does NOT include Unicode normalization
    # These would be normalized differently under NFC/NFKC but should remain unchanged
    test_cases = [
        ("ﬁ", "ﬁ"),                # U+FB01 ligature, should not be normalized to 'fi'
        ("²", "²"),                  # Superscript 2, should not be normalized to '2'
        ("Ω", "Ω"),                  # Ohm symbol, should not be normalized to Omega
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_no_whitespace_collapsing():
    """Test that whitespace is NOT collapsed (BFCL-exact behavior)."""
    # BFCL v4 spec does NOT include whitespace normalization
    test_cases = [
        ("hello   world", "hello   world"),  # Multiple spaces preserved
        ("hello\tworld", "hello\tworld"),      # Tabs preserved
        ("hello\nworld", "hello\nworld"),      # Newlines preserved
        ("  hello  world  ", "  hello  world  "),  # Leading/trailing spaces preserved
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_combined():
    """Test combinations of lowercase and punctuation removal."""
    test_cases = [
        ("Hello, World!", "hello world!"),  # Lowercase + punctuation (! preserved)
        ("PATH/TO/FILE.TXT", "pathtofiletxt"),  # Lowercase + multiple punctuation
        ("User_Name_123", "username123"),  # Lowercase + underscores
        ("(2*2)^2=16", "222=16"),        # Multiple punctuation types (= preserved)
        ('"quoted" text', 'quoted text'),  # Quotes + spaces
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_empty_string():
    """Test normalization of empty and edge-case strings."""
    test_cases = [
        ("", ""),                    # Empty string
        ("   ", "   "),            # Only whitespace (preserved)
        (",./*-^()", ""),          # Only punctuation
        ("A", "a"),                # Single character
        ("a", "a"),                # Single lowercase character
        ("1", "1"),                # Single digit
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_normalization_unsupported_punctuation():
    """Test that punctuation not in BFCL spec is preserved."""
    # BFCL v4 only removes: , . / - _ * ^ ( ) and quotes
    # Other punctuation should be preserved
    test_cases = [
        ("hello:world", "hello:world"),      # Colon preserved
        ("hello;world", "hello;world"),      # Semicolon preserved
        ("hello?world", "hello?world"),      # Question mark preserved
        ("hello!world", "hello!world"),      # Exclamation preserved
        ("hello<world", "hello<world"),      # Less than preserved
        ("hello>world", "hello>world"),      # Greater than preserved
        ("hello@world", "hello@world"),      # At symbol preserved
        ("hello#world", "hello#world"),      # Hash preserved
        ("hello$world", "hello$world"),      # Dollar preserved
        ("hello%world", "hello%world"),      # Percent preserved
        ("hello&world", "hello&world"),      # Ampersand preserved
        ("hello+world", "hello+world"),      # Plus preserved
        ("hello=world", "hello=world"),      # Equals preserved
        ("hello|world", "hello|world"),      # Pipe preserved
        ("hello\\world", "hello\\world"),    # Backslash preserved
        ("hello~world", "hello~world"),      # Tilde preserved
        ("hello`world", "hello`world"),      # Backtick preserved
    ]
    
    for input_text, expected in test_cases:
        result = _normalize_answer(input_text)
        assert result == expected, f"Expected '{expected}' but got '{result}' for input '{input_text}'"


def test_v4_environment_uses_normalization():
    """Test that the environment uses the BFCL-exact normalization function."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "What is 2+2?"}]], 
        "answer": ['{"answer": "Four"}'],  # Model answers with "Four" - as list with one element
        "info": [{}]
    })
    env = BFCLV4SingleTurnEnv(dataset=ds)
    
    # Create a completion that would match after normalization
    completion = [{"role": "assistant", "content": '{"answer": "four"}'}]
    
    # The environment should normalize both answers and find a match
    success = env._determine_v4_success([], completion, 'Four', {}, {})
    assert success is True, "Environment should use BFCL-exact normalization"


def test_v4_environment_normalization_mismatch():
    """Test that environment correctly fails when normalization doesn't match."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "What is 2+2?"}]], 
        "answer": ['{"answer": "Four"}'],  # Gold answer - as list with one element
        "info": [{}]
    })
    env = BFCLV4SingleTurnEnv(dataset=ds)
    
    # Create a completion that doesn't match even after normalization
    completion = [{"role": "assistant", "content": '{"answer": "Five"}'}]
    
    # The environment should normalize both answers and find no match
    success = env._determine_v4_success([], completion, 'Four', {}, {})
    assert success is False, "Environment should correctly fail when normalization doesn't match"


def test_v4_environment_json_parsing():
    """Test that environment correctly parses JSON and extracts answer field."""
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "What is 2+2?"}]], 
        "answer": ["4"],  # Gold answer - as list with one element
        "info": [{}]
    })
    env = BFCLV4SingleTurnEnv(dataset=ds)
    
    # Test valid JSON with answer field
    completion = [{"role": "assistant", "content": '{"answer": "4"}'}]
    success = env._determine_v4_success([], completion, "4", {}, {})
    assert success is True, "Environment should parse valid JSON and match answer"
    
    # Test JSON without answer field
    completion = [{"role": "assistant", "content": '{"result": "4"}'}]
    success = env._determine_v4_success([], completion, "4", {}, {})
    assert success is False, "Environment should fail when answer field is missing"
    
    # Test invalid JSON
    completion = [{"role": "assistant", "content": 'invalid json'}]
    success = env._determine_v4_success([], completion, "4", {}, {})
    assert success is False, "Environment should fail when JSON is invalid"

