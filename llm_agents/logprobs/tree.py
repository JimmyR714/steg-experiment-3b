"""Probability tree exploration for LLM continuations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator

from llm_agents.models.base import BaseModel


@dataclass
class TreeNode:
    """A node in a probability tree representing a token continuation.

    Attributes:
        token: The token string at this node (empty string for root).
        logprob: The log-probability of this token given its prefix.
        children: Child nodes representing further continuations.
        cumulative_logprob: Sum of log-probs from root to this node.
    """

    token: str
    logprob: float
    children: list[TreeNode] = field(default_factory=list)
    cumulative_logprob: float = 0.0


def build_prob_tree(
    model: BaseModel,
    prompt: str,
    *,
    branch_factor: int = 3,
    depth: int = 3,
) -> TreeNode:
    """Build a tree of the most probable continuations.

    At each node, queries the model for the top `branch_factor` tokens,
    then recurses with each token appended to the prompt up to `depth`.

    Args:
        model: An LLM model that supports get_logprobs.
        prompt: The initial prompt to explore continuations for.
        branch_factor: Number of top tokens to branch on at each level.
        depth: Maximum depth of the tree.

    Returns:
        A root TreeNode with token="" representing the prompt, whose
        children form the exploration tree.
    """
    root = TreeNode(token="", logprob=0.0, cumulative_logprob=0.0)
    _expand_node(model, prompt, root, branch_factor, depth)
    return root


def _expand_node(
    model: BaseModel,
    prompt: str,
    node: TreeNode,
    branch_factor: int,
    depth: int,
) -> None:
    """Recursively expand a tree node by querying the model."""
    if depth <= 0:
        return

    result = model.get_logprobs(prompt, max_tokens=1, top_k=branch_factor)

    if not result.top_k_per_position:
        return

    top_tokens = result.top_k_per_position[0][:branch_factor]

    for token_lp in top_tokens:
        child = TreeNode(
            token=token_lp.token,
            logprob=token_lp.logprob,
            cumulative_logprob=node.cumulative_logprob + token_lp.logprob,
        )
        node.children.append(child)
        _expand_node(
            model, prompt + token_lp.token, child, branch_factor, depth - 1
        )


def print_tree(node: TreeNode, indent: int = 0) -> None:
    """Pretty-print the probability tree to stdout.

    Args:
        node: The tree node to print.
        indent: Current indentation level.
    """
    label = node.token if node.token else "<root>"
    print(
        f"{'  ' * indent}{label} "
        f"(lp={node.logprob:.4f}, cum={node.cumulative_logprob:.4f})"
    )
    for child in node.children:
        print_tree(child, indent + 1)


def tree_to_dict(node: TreeNode) -> dict[str, Any]:
    """Serialize a tree node to a nested dictionary.

    Args:
        node: The tree node to serialize.

    Returns:
        A JSON-friendly nested dict representation.
    """
    return {
        "token": node.token,
        "logprob": node.logprob,
        "cumulative_logprob": node.cumulative_logprob,
        "children": [tree_to_dict(c) for c in node.children],
    }


def best_path(node: TreeNode) -> list[TreeNode]:
    """Return the highest cumulative-probability root-to-leaf path.

    Args:
        node: The root node to search from.

    Returns:
        A list of TreeNodes from root to the leaf with the highest
        cumulative log-probability.
    """
    if not node.children:
        return [node]

    best_child_path: list[TreeNode] = []
    best_cum_lp = float("-inf")

    for child in node.children:
        child_path = best_path(child)
        leaf_cum_lp = child_path[-1].cumulative_logprob
        if leaf_cum_lp > best_cum_lp:
            best_cum_lp = leaf_cum_lp
            best_child_path = child_path

    return [node] + best_child_path


def all_paths(node: TreeNode) -> Generator[list[TreeNode], None, None]:
    """Yield all root-to-leaf paths with their cumulative log-probs.

    Args:
        node: The root node to traverse.

    Yields:
        Lists of TreeNodes representing each root-to-leaf path.
    """
    if not node.children:
        yield [node]
        return

    for child in node.children:
        for path in all_paths(child):
            yield [node] + path
