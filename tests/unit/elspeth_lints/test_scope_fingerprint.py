# test_scope_fingerprint.py
import ast

import pytest

from elspeth_lints.rules.trust_tier.tier_model.scope_fingerprint import (
    compute_scope_fingerprint,
    enclosing_scope_node,
)


def _func(src: str) -> ast.AST:
    """Return the first FunctionDef/AsyncFunctionDef/ClassDef in src."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return node
    raise AssertionError("no scope node in source")


def test_hash_is_64_char_lowercase_hex() -> None:
    fp = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    assert len(fp) == 64
    assert fp == fp.lower()
    bytes.fromhex(fp)  # raises if not hex


def test_reformatting_and_comments_are_free() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x):  # a comment\n        return x.get('a')\n"))
    assert a == b


def test_docstring_edit_is_free() -> None:
    a = compute_scope_fingerprint(_func('def f(x):\n    "old doc"\n    return x.get("a")\n'))
    b = compute_scope_fingerprint(_func('def f(x):\n    "completely different doc"\n    return x.get("a")\n'))
    assert a == b


def test_adding_a_docstring_is_free() -> None:
    a = compute_scope_fingerprint(_func('def f(x):\n    return x.get("a")\n'))
    b = compute_scope_fingerprint(_func('def f(x):\n    "now documented"\n    return x.get("a")\n'))
    assert a == b


def test_body_change_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x):\n    return x.get('b')\n"))
    assert a != b


def test_parameter_change_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x, y):\n    return x.get('a')\n"))
    assert a != b


def test_decorator_change_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("@retry\ndef f(x):\n    return x.get('a')\n"))
    assert a != b


def test_rename_changes_hash() -> None:
    a = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def g(x):\n    return x.get('a')\n"))
    assert a != b


def test_class_scope_first_string_stmt_is_treated_as_docstring() -> None:
    # The class docstring is dropped; a method body change is not.
    a = compute_scope_fingerprint(_func('class C:\n    "doc"\n    def m(self):\n        return 1\n'))
    b = compute_scope_fingerprint(_func('class C:\n    "other"\n    def m(self):\n        return 1\n'))
    assert a == b


def test_module_level_fallback_uses_whole_module() -> None:
    tree = ast.parse("import os\nX = os.environ.get('A')\n")
    fp_a = compute_scope_fingerprint(None, module=tree)
    fp_b = compute_scope_fingerprint(None, module=ast.parse("import os\nX = os.environ.get('B')\n"))
    assert len(fp_a) == 64
    assert fp_a != fp_b


def test_module_fallback_requires_module_arg() -> None:
    with pytest.raises(ValueError, match="module"):
        compute_scope_fingerprint(None)


def test_enclosing_scope_node_finds_innermost_def() -> None:
    src = "class C:\n    def m(self, x):\n        return x.get('a')\n"
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    # ancestors innermost-first: the Call's enclosing scope is method m, not class C.
    ancestors = _ancestors_of(tree, call)
    scope = enclosing_scope_node(ancestors)
    assert isinstance(scope, ast.FunctionDef)
    assert scope.name == "m"


def test_async_function_is_a_scope_and_differs_from_sync() -> None:
    a = compute_scope_fingerprint(_func("async def f(x):\n    return x.get('a')\n"))
    b = compute_scope_fingerprint(_func("def f(x):\n    return x.get('a')\n"))
    assert len(a) == 64
    assert a != b  # async-ness is part of the scope identity (rule 1 lists AsyncFunctionDef)


def test_innermost_nested_function_is_the_scope() -> None:
    src = "def outer():\n    def inner(x):\n        return x.get('a')\n    return inner\n"
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    ancestors = _ancestors_of(tree, call)
    scope = enclosing_scope_node(ancestors)
    assert isinstance(scope, ast.FunctionDef)
    assert scope.name == "inner"  # innermost def, not the outer wrapper


def test_docstring_only_body_does_not_collide_with_pass() -> None:
    # A function whose body is only a docstring (empty after strip) must not
    # hash-collide with a different-bodied function.
    a = compute_scope_fingerprint(_func('def f():\n    "just a doc"\n'))
    b = compute_scope_fingerprint(_func("def f():\n    pass\n"))
    assert a != b


def test_module_fallback_ignores_type_ignore_comments() -> None:
    # Module-path symmetry with the scope path (Fix 1): a ``# type: ignore`` is a
    # comment (rule 4), not code the judge reasoned about, so it must be free even
    # when the source was parsed with type_comments=True.
    plain = ast.parse("import os\nX = os.environ['A']\n")
    with_ignore = ast.parse("import os\nX = os.environ['A']  # type: ignore\n", type_comments=True)
    assert compute_scope_fingerprint(None, module=plain) == compute_scope_fingerprint(None, module=with_ignore)


def _ancestors_of(tree: ast.AST, target: ast.AST) -> list[ast.AST]:
    """Return [target, parent, ..., module] — innermost first."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    chain: list[ast.AST] = [target]
    cur = target
    while id(cur) in parents:
        cur = parents[id(cur)]
        chain.append(cur)
    return chain
