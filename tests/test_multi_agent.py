"""Tests for multi-agent communication (Phase 7)."""

from __future__ import annotations

import json

import pytest

from llm_agents.agents.agent import Agent, AgentResponse
from llm_agents.agents.message_bus import Message, MessageBus
from llm_agents.agents.multi_agent import MultiAgentSystem
from llm_agents.agents.task import TaskResult
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult, TokenLogProb
from llm_agents.tools.base import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel(BaseModel):
    """A mock model that returns pre-configured responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return CompletionResult(text=text)

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        return LogProbResult(prompt=prompt)


def _tool_call_json(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments})


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------


class TestMessage:
    def test_basic_construction(self):
        msg = Message(sender="alice", recipient="bob", content="hello")
        assert msg.sender == "alice"
        assert msg.recipient == "bob"
        assert msg.content == "hello"
        assert msg.metadata == {}

    def test_with_metadata(self):
        msg = Message(
            sender="a", recipient="b", content="x", metadata={"priority": 1}
        )
        assert msg.metadata["priority"] == 1

    def test_broadcast_recipient(self):
        msg = Message(sender="a", recipient="*", content="hi all")
        assert msg.recipient == "*"


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------


class TestMessageBus:
    def test_subscribe_and_list(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        assert bus.subscribers == ["alice", "bob"]

    def test_unsubscribe(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.unsubscribe("alice")
        assert bus.subscribers == []

    def test_send_direct_message(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        msg = Message(sender="alice", recipient="bob", content="hi bob")
        bus.send(msg)
        received = bus.receive("bob")
        assert len(received) == 1
        assert received[0].content == "hi bob"
        assert received[0].sender == "alice"

    def test_receive_clears_queue(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        bus.send(Message(sender="alice", recipient="bob", content="msg1"))
        assert len(bus.receive("bob")) == 1
        assert len(bus.receive("bob")) == 0

    def test_peek_does_not_clear(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        bus.send(Message(sender="alice", recipient="bob", content="msg"))
        assert len(bus.peek("bob")) == 1
        assert len(bus.peek("bob")) == 1  # still there

    def test_broadcast_delivers_to_all_except_sender(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        bus.subscribe("charlie")
        bus.send(Message(sender="alice", recipient="*", content="hello all"))
        assert len(bus.receive("bob")) == 1
        assert len(bus.receive("charlie")) == 1
        assert len(bus.receive("alice")) == 0  # sender excluded

    def test_send_to_unknown_recipient_raises(self):
        bus = MessageBus()
        bus.subscribe("alice")
        with pytest.raises(ValueError, match="Unknown recipient"):
            bus.send(Message(sender="alice", recipient="ghost", content="x"))

    def test_all_messages_tracks_history(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        bus.send(Message(sender="alice", recipient="bob", content="m1"))
        bus.send(Message(sender="bob", recipient="alice", content="m2"))
        assert len(bus.all_messages) == 2
        assert bus.all_messages[0].content == "m1"

    def test_multiple_messages_queued(self):
        bus = MessageBus()
        bus.subscribe("alice")
        bus.subscribe("bob")
        bus.send(Message(sender="alice", recipient="bob", content="first"))
        bus.send(Message(sender="alice", recipient="bob", content="second"))
        received = bus.receive("bob")
        assert len(received) == 2
        assert received[0].content == "first"
        assert received[1].content == "second"


# ---------------------------------------------------------------------------
# TaskResult dataclass
# ---------------------------------------------------------------------------


class TestTaskResult:
    def test_basic_construction(self):
        result = TaskResult(result="done")
        assert result.result == "done"
        assert result.agent_trace == []
        assert result.logprobs is None

    def test_with_trace_and_logprobs(self):
        msg = Message(sender="a", recipient="b", content="x")
        token = TokenLogProb(token="hi", logprob=-0.5, rank=0)
        lp = LogProbResult(prompt="test", tokens=[token])
        result = TaskResult(result="ok", agent_trace=[msg], logprobs=lp)
        assert len(result.agent_trace) == 1
        assert result.logprobs is not None


# ---------------------------------------------------------------------------
# MultiAgentSystem — communication tools injection
# ---------------------------------------------------------------------------


class TestMultiAgentToolInjection:
    def test_communication_tools_injected(self):
        model = MockModel(["hello"])
        agent = Agent(name="alice", model=model, enable_cot=False)
        bus = MessageBus()
        MultiAgentSystem(agents=[agent], message_bus=bus)
        tool_names = [t.name for t in agent._registry.list_tools()]
        assert "send_message" in tool_names
        assert "broadcast" in tool_names
        assert "wait_for_reply" in tool_names
        assert "list_agents" in tool_names

    def test_existing_tools_preserved(self):
        model = MockModel(["hello"])
        custom_tool = Tool(
            name="custom",
            description="A custom tool.",
            parameters_schema={"type": "object", "properties": {}},
            fn=lambda: "custom result",
        )
        agent = Agent(name="alice", model=model, tools=[custom_tool], enable_cot=False)
        bus = MessageBus()
        MultiAgentSystem(agents=[agent], message_bus=bus)
        tool_names = [t.name for t in agent._registry.list_tools()]
        assert "custom" in tool_names
        assert "send_message" in tool_names

    def test_agents_subscribed_to_bus(self):
        model = MockModel(["x"])
        a1 = Agent(name="alice", model=model, enable_cot=False)
        a2 = Agent(name="bob", model=model, enable_cot=False)
        bus = MessageBus()
        MultiAgentSystem(agents=[a1, a2], message_bus=bus)
        assert "alice" in bus.subscribers
        assert "bob" in bus.subscribers


# ---------------------------------------------------------------------------
# MultiAgentSystem — communication tool functionality
# ---------------------------------------------------------------------------


class TestCommunicationTools:
    def _setup_system(self, alice_responses, bob_responses):
        alice_model = MockModel(alice_responses)
        bob_model = MockModel(bob_responses)
        alice = Agent(name="alice", model=alice_model, enable_cot=False)
        bob = Agent(name="bob", model=bob_model, enable_cot=False)
        bus = MessageBus()
        system = MultiAgentSystem(agents=[alice, bob], message_bus=bus)
        return system, alice, bob, bus

    def test_send_message_tool(self):
        """Coordinator sends a message to another agent via tool call."""
        send_call = _tool_call_json(
            "send_message", {"to": "bob", "content": "do task X"}
        )
        system, alice, bob, bus = self._setup_system(
            alice_responses=[send_call, "Task complete."],
            bob_responses=["I did task X."],
        )
        result = system.run_task("Please delegate to bob.", coordinator="alice")
        assert result.result == "Task complete."
        assert len(result.agent_trace) > 0
        # Bob should have received and processed the message.
        sent_msgs = [m for m in result.agent_trace if m.recipient == "bob"]
        assert len(sent_msgs) >= 1

    def test_broadcast_tool(self):
        """Coordinator broadcasts to all agents."""
        broadcast_call = _tool_call_json(
            "broadcast", {"content": "attention everyone"}
        )
        alice_model = MockModel([broadcast_call, "Done broadcasting."])
        bob_model = MockModel(["Acknowledged."])
        charlie_model = MockModel(["Got it."])
        alice = Agent(name="alice", model=alice_model, enable_cot=False)
        bob = Agent(name="bob", model=bob_model, enable_cot=False)
        charlie = Agent(name="charlie", model=charlie_model, enable_cot=False)
        bus = MessageBus()
        system = MultiAgentSystem(
            agents=[alice, bob, charlie], message_bus=bus
        )
        result = system.run_task("Broadcast a message.", coordinator="alice")
        assert result.result == "Done broadcasting."
        # Both bob and charlie should have gotten the broadcast.
        broadcast_msgs = [m for m in bus.all_messages if m.recipient == "*"]
        assert len(broadcast_msgs) == 1

    def test_list_agents_tool(self):
        """Agent can list all agents in the system."""
        list_call = _tool_call_json("list_agents", {})
        system, alice, bob, bus = self._setup_system(
            alice_responses=[list_call, "I see the agents."],
            bob_responses=["ok"],
        )
        result = system.run_task("Who is available?", coordinator="alice")
        assert result.result == "I see the agents."

    def test_wait_for_reply_no_messages(self):
        """wait_for_reply returns a 'no reply' message when queue is empty."""
        wait_call = _tool_call_json(
            "wait_for_reply", {"from_agent": "bob"}
        )
        system, alice, bob, bus = self._setup_system(
            alice_responses=[wait_call, "No reply yet."],
            bob_responses=["ok"],
        )
        result = system.run_task("Wait for bob.", coordinator="alice")
        assert result.result == "No reply yet."

    def test_send_to_unknown_agent(self):
        """send_message to non-existent agent returns error string."""
        send_call = _tool_call_json(
            "send_message", {"to": "ghost", "content": "hello"}
        )
        system, alice, bob, bus = self._setup_system(
            alice_responses=[send_call, "Got an error."],
            bob_responses=["ok"],
        )
        result = system.run_task("Send to ghost.", coordinator="alice")
        assert result.result == "Got an error."


# ---------------------------------------------------------------------------
# MultiAgentSystem — run_task
# ---------------------------------------------------------------------------


class TestRunTask:
    def test_unknown_coordinator_raises(self):
        model = MockModel(["x"])
        agent = Agent(name="alice", model=model, enable_cot=False)
        bus = MessageBus()
        system = MultiAgentSystem(agents=[agent], message_bus=bus)
        with pytest.raises(KeyError, match="Unknown coordinator"):
            system.run_task("do something", coordinator="nobody")

    def test_simple_task_no_delegation(self):
        """Coordinator answers directly without using communication tools."""
        model = MockModel(["The answer is 42."])
        agent = Agent(name="alice", model=model, enable_cot=False)
        bus = MessageBus()
        system = MultiAgentSystem(agents=[agent], message_bus=bus)
        result = system.run_task("What is the answer?", coordinator="alice")
        assert result.result == "The answer is 42."
        assert isinstance(result, TaskResult)

    def test_task_result_contains_trace(self):
        """Verify agent_trace captures messages."""
        send_call = _tool_call_json(
            "send_message", {"to": "bob", "content": "help me"}
        )
        alice_model = MockModel([send_call, "All done."])
        bob_model = MockModel(["I helped."])
        alice = Agent(name="alice", model=alice_model, enable_cot=False)
        bob = Agent(name="bob", model=bob_model, enable_cot=False)
        bus = MessageBus()
        system = MultiAgentSystem(agents=[alice, bob], message_bus=bus)
        result = system.run_task("Need help.", coordinator="alice")
        assert result.result == "All done."
        assert len(result.agent_trace) >= 1
        assert any(m.sender == "alice" for m in result.agent_trace)

    def test_agents_property(self):
        model = MockModel(["x"])
        a1 = Agent(name="alice", model=model, enable_cot=False)
        a2 = Agent(name="bob", model=model, enable_cot=False)
        system = MultiAgentSystem(agents=[a1, a2])
        agents = system.agents
        assert "alice" in agents
        assert "bob" in agents
        assert len(agents) == 2

    def test_default_bus_created(self):
        """If no bus is provided, one is created automatically."""
        model = MockModel(["ok"])
        agent = Agent(name="alice", model=model, enable_cot=False)
        system = MultiAgentSystem(agents=[agent])
        assert system.bus is not None
        assert "alice" in system.bus.subscribers
