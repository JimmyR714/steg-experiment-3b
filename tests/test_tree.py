"""Tests for probability tree exploration using a mock model."""

from __future__ import annotations

import math

import pytest

from llm_agents.logprobs.tree import (
    TreeNode,
    all_paths,
    best_path,
    build_prob_tree,
    print_tree,
    tree_to_dict,
)
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb


# -----------------------------------------------------------------------
# Mock model that returns deterministic top-k tokens
# -----------------------------------------------------------------------


class MockModel(BaseModel):
    """A mock model that returns pre-configured top-k tokens.

    For any prompt, returns a fixed set of top tokens. The token text
    and log-probs can be customized per call via the token_map.
    """

    def __init__(self, token_map: dict[str, list[tuple[str, float]]] | None = None):
        """Initialize with an optional prompt -> tokens mapping.

        Args:
            token_map: Maps prompt strings to lists of (token, logprob) pairs.
                       If a prompt is not found, returns a default set.
        """
        self._token_map = token_map or {}
        self._default_tokens = [("a", -0.5), ("b", -1.0), ("c", -1.5)]

    def generate(self, prompt: str, **kwargs) -> CompletionResult:
        return CompletionResult(text="mock")

    def get_logprobs(self, prompt: str, **kwargs) -> LogProbResult:
        top_k = kwargs.get("top_k", 5)
        tokens = self._token_map.get(prompt, self._default_tokens)[:top_k]

        top_k_list = [
            TokenLogProb(token=t, logprob=lp, rank=i)
            for i, (t, lp) in enumerate(tokens)
        ]
        chosen = top_k_list[0] if top_k_list else None

        return LogProbResult(
            prompt=prompt,
            tokens=[chosen] if chosen else [],
            top_k_per_position=[top_k_list] if top_k_list else [],
        )


# -----------------------------------------------------------------------
# TreeNode dataclass tests
# -----------------------------------------------------------------------


class TestTreeNode:
    def test_defaults(self):
        node = TreeNode(token="x", logprob=-1.0)
        assert node.token == "x"
        assert node.logprob == -1.0
        assert node.children == []
        assert node.cumulative_logprob == 0.0

    def test_with_children(self):
        child = TreeNode(token="y", logprob=-0.5, cumulative_logprob=-0.5)
        parent = TreeNode(token="", logprob=0.0, children=[child])
        assert len(parent.children) == 1
        assert parent.children[0] is child


# -----------------------------------------------------------------------
# build_prob_tree tests
# -----------------------------------------------------------------------


class TestBuildProbTree:
    def test_root_structure(self):
        model = MockModel()
        root = build_prob_tree(model, "Hello", branch_factor=2, depth=1)

        assert root.token == ""
        assert root.logprob == 0.0
        assert root.cumulative_logprob == 0.0
        assert len(root.children) == 2

    def test_depth_one(self):
        model = MockModel()
        root = build_prob_tree(model, "Hello", branch_factor=3, depth=1)

        assert len(root.children) == 3
        assert root.children[0].token == "a"
        assert root.children[0].logprob == pytest.approx(-0.5)
        assert root.children[0].cumulative_logprob == pytest.approx(-0.5)
        # Leaves should have no children at depth=1
        for child in root.children:
            assert child.children == []

    def test_depth_two(self):
        model = MockModel()
        root = build_prob_tree(model, "X", branch_factor=2, depth=2)

        assert len(root.children) == 2
        for child in root.children:
            assert len(child.children) == 2
            for grandchild in child.children:
                assert grandchild.children == []

    def test_cumulative_logprobs(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=2, depth=2)

        # First child: token "a" with lp -0.5
        first_child = root.children[0]
        assert first_child.cumulative_logprob == pytest.approx(-0.5)

        # Grandchild through "a" -> "a": cumulative = -0.5 + -0.5 = -1.0
        grandchild = first_child.children[0]
        assert grandchild.cumulative_logprob == pytest.approx(-1.0)

    def test_depth_zero(self):
        model = MockModel()
        root = build_prob_tree(model, "Hello", branch_factor=3, depth=0)
        assert root.children == []

    def test_branch_factor_one(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=1, depth=3)

        # Should be a single chain
        node = root
        for _ in range(3):
            assert len(node.children) == 1
            node = node.children[0]
        assert node.children == []

    def test_custom_token_map(self):
        token_map = {
            "Start": [("X", -0.2), ("Y", -0.8)],
            "StartX": [("1", -0.3), ("2", -0.7)],
            "StartY": [("3", -0.4), ("4", -0.9)],
        }
        model = MockModel(token_map=token_map)
        root = build_prob_tree(model, "Start", branch_factor=2, depth=2)

        assert root.children[0].token == "X"
        assert root.children[1].token == "Y"
        assert root.children[0].children[0].token == "1"
        assert root.children[0].children[1].token == "2"
        assert root.children[1].children[0].token == "3"


# -----------------------------------------------------------------------
# print_tree tests
# -----------------------------------------------------------------------


class TestPrintTree:
    def test_prints_root(self, capsys):
        root = TreeNode(token="", logprob=0.0)
        print_tree(root)
        output = capsys.readouterr().out
        assert "<root>" in output
        assert "lp=0.0000" in output

    def test_prints_children(self, capsys):
        child = TreeNode(token="hello", logprob=-0.5, cumulative_logprob=-0.5)
        root = TreeNode(token="", logprob=0.0, children=[child])
        print_tree(root)
        output = capsys.readouterr().out
        assert "hello" in output
        assert "<root>" in output

    def test_indentation(self, capsys):
        grandchild = TreeNode(token="gc", logprob=-1.0, cumulative_logprob=-1.5)
        child = TreeNode(
            token="ch", logprob=-0.5, cumulative_logprob=-0.5, children=[grandchild]
        )
        root = TreeNode(token="", logprob=0.0, children=[child])
        print_tree(root)
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 3
        # Grandchild should be indented more than child
        assert lines[2].startswith("    ")  # 2 levels of indent


# -----------------------------------------------------------------------
# tree_to_dict tests
# -----------------------------------------------------------------------


class TestTreeToDict:
    def test_leaf_node(self):
        node = TreeNode(token="x", logprob=-1.0, cumulative_logprob=-1.0)
        d = tree_to_dict(node)
        assert d == {
            "token": "x",
            "logprob": -1.0,
            "cumulative_logprob": -1.0,
            "children": [],
        }

    def test_nested_structure(self):
        child = TreeNode(token="b", logprob=-0.5, cumulative_logprob=-0.5)
        root = TreeNode(token="", logprob=0.0, children=[child])
        d = tree_to_dict(root)

        assert d["token"] == ""
        assert len(d["children"]) == 1
        assert d["children"][0]["token"] == "b"
        assert d["children"][0]["children"] == []

    def test_json_serializable(self):
        import json

        model = MockModel()
        root = build_prob_tree(model, "Test", branch_factor=2, depth=2)
        d = tree_to_dict(root)
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# -----------------------------------------------------------------------
# best_path tests
# -----------------------------------------------------------------------


class TestBestPath:
    def test_single_node(self):
        root = TreeNode(token="", logprob=0.0)
        path = best_path(root)
        assert len(path) == 1
        assert path[0] is root

    def test_chooses_highest_cumulative(self):
        # Build a small tree manually
        leaf_good = TreeNode(token="g", logprob=-0.1, cumulative_logprob=-0.3)
        leaf_bad = TreeNode(token="b", logprob=-2.0, cumulative_logprob=-2.5)
        child_a = TreeNode(
            token="a", logprob=-0.2, cumulative_logprob=-0.2, children=[leaf_good]
        )
        child_b = TreeNode(
            token="b", logprob=-0.5, cumulative_logprob=-0.5, children=[leaf_bad]
        )
        root = TreeNode(token="", logprob=0.0, children=[child_a, child_b])

        path = best_path(root)
        tokens = [n.token for n in path]
        assert tokens == ["", "a", "g"]

    def test_with_mock_model(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=2, depth=2)
        path = best_path(root)

        # The best path should always pick the first token ("a", lp=-0.5)
        # which has the highest logprob at each level
        assert len(path) == 3  # root + 2 levels
        assert path[0] is root
        assert path[1].token == "a"
        assert path[2].token == "a"
        assert path[-1].cumulative_logprob == pytest.approx(-1.0)


# -----------------------------------------------------------------------
# all_paths tests
# -----------------------------------------------------------------------


class TestAllPaths:
    def test_single_node(self):
        root = TreeNode(token="", logprob=0.0)
        paths = list(all_paths(root))
        assert len(paths) == 1
        assert len(paths[0]) == 1

    def test_counts_match_tree(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=2, depth=2)
        paths = list(all_paths(root))
        # With branch_factor=2 and depth=2, there should be 2^2 = 4 leaf paths
        assert len(paths) == 4

    def test_all_paths_start_at_root(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=2, depth=2)
        for path in all_paths(root):
            assert path[0] is root

    def test_all_paths_end_at_leaves(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=2, depth=2)
        for path in all_paths(root):
            assert path[-1].children == []

    def test_path_length(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=3, depth=3)
        for path in all_paths(root):
            assert len(path) == 4  # root + 3 depth levels

    def test_branch_factor_three_depth_two(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=3, depth=2)
        paths = list(all_paths(root))
        # 3^2 = 9 paths
        assert len(paths) == 9

    def test_best_path_in_all_paths(self):
        model = MockModel()
        root = build_prob_tree(model, "P", branch_factor=2, depth=2)
        bp = best_path(root)
        ap = list(all_paths(root))

        # best_path result should match one of all_paths (by node identity)
        bp_tokens = [n.token for n in bp]
        found = any(
            [n.token for n in p] == bp_tokens for p in ap
        )
        assert found
