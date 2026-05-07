"""
BaseAgent: foundation class for every agent in the fund.

Each agent has:
  - A system prompt encoding its role, mandate, and personality
  - A Claude API client (claude-sonnet-4-20250514)
  - A ChromaDB vector memory store
  - A SQLite message-bus inbox/outbox
  - Tool access gated by role
  - Structured output parsing via Pydantic

Agents are designed to be called synchronously or scheduled asynchronously.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel

from hedge_fund.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, AUDIT_DB_PATH
from hedge_fund.memory.vector_store import AgentMemory
from hedge_fund.messaging import bus as message_bus
from hedge_fund.models.schemas import AgentMessage, MessagePriority

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """
    Core agent class. Subclass this to create specific agents.

    Subclasses must define:
        agent_id: str         — unique identifier (e.g. "cio", "pm_longshort")
        role_name: str        — human-readable role name
        system_prompt: str    — full system prompt for the LLM

    Optionally override:
        allowed_tools: list   — Claude API tool specs this agent can use
        max_context_messages  — how many recent messages to keep in context
    """

    agent_id: str = "base_agent"
    role_name: str = "Base Agent"
    system_prompt: str = "You are an AI assistant."
    allowed_tools: list[dict] = []
    max_context_messages: int = 20

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        self.db_path = db_path
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.memory = AgentMemory(self.agent_id)
        self._message_history: list[dict] = []  # short-term context
        self._start_time = datetime.utcnow()
        self._task_count = 0
        self._error_count = 0
        self._last_response_time: float = 0.0
        logger.info(f"[{self.agent_id}] Initialized")

    # ── Messaging ─────────────────────────────────────────────────────────────

    def send_message(
        self,
        recipient: str,
        subject: str,
        body: str,
        payload: Optional[dict] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        thread_id: Optional[str] = None,
        requires_reply: bool = False,
    ) -> str:
        """Send a message to another agent. Returns message_id."""
        msg_id = message_bus.send(
            sender=self.agent_id,
            recipient=recipient,
            subject=subject,
            body=body,
            payload=payload,
            priority=priority,
            thread_id=thread_id,
            requires_reply=requires_reply,
            db_path=self.db_path,
        )
        logger.debug(f"[{self.agent_id}] → [{recipient}] {subject} (id={msg_id[:8]})")
        return msg_id

    def receive_messages(self, limit: int = 10) -> list[AgentMessage]:
        """Pull unread messages from this agent's inbox."""
        messages = message_bus.receive(
            agent_id=self.agent_id,
            limit=limit,
            mark_read=True,
            db_path=self.db_path,
        )
        if messages:
            logger.debug(f"[{self.agent_id}] Received {len(messages)} messages")
        return messages

    def get_queue_depth(self) -> int:
        return message_bus.queue_depth(self.agent_id, db_path=self.db_path)

    # ── LLM Core ─────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        user_message: str,
        additional_context: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Send a message to Claude and return the text response.
        Maintains a rolling short-term context window.
        """
        # Retrieve relevant memories
        memories = self.memory.search(user_message[:200], n_results=3)
        memory_context = ""
        if memories:
            memory_context = "\n\n[Relevant past context from memory]\n" + "\n---\n".join(
                m["text"][:300] for m in memories
            )

        full_user_msg = user_message
        if additional_context:
            full_user_msg = f"{additional_context}\n\n{user_message}"
        if memory_context:
            full_user_msg = f"{full_user_msg}{memory_context}"

        # Add to rolling context
        self._message_history.append({"role": "user", "content": full_user_msg})
        # Keep context within limits
        if len(self._message_history) > self.max_context_messages * 2:
            self._message_history = self._message_history[-(self.max_context_messages * 2):]

        kwargs: dict[str, Any] = {
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "system": self.system_prompt,
            "messages": self._message_history,
        }
        if self.allowed_tools:
            kwargs["tools"] = self.allowed_tools

        t0 = time.time()
        try:
            response = self._client.messages.create(**kwargs)
            self._last_response_time = time.time() - t0
            self._task_count += 1

            # Handle tool use blocks if present
            assistant_text = self._process_response(response, kwargs)

            self._message_history.append({"role": "assistant", "content": assistant_text})
            return assistant_text

        except anthropic.RateLimitError:
            logger.warning(f"[{self.agent_id}] Rate limit hit, waiting 30s")
            time.sleep(30)
            return self._call_llm(user_message, additional_context)
        except Exception as e:
            self._error_count += 1
            logger.error(f"[{self.agent_id}] LLM error: {e}")
            raise

    def _process_response(self, response, original_kwargs: dict) -> str:
        """
        Process a Claude response, handling tool use if present.
        Returns the final text content.
        """
        # Collect all content blocks
        text_blocks = []
        tool_use_blocks = []

        for block in response.content:
            if hasattr(block, "text"):
                text_blocks.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        if not tool_use_blocks:
            return "\n".join(text_blocks)

        # Execute tools and get final response
        tool_results = []
        for tb in tool_use_blocks:
            result = self._execute_tool(tb.name, tb.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": str(result),
            })

        # Feed results back to Claude
        messages = original_kwargs["messages"] + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        followup_kwargs = {**original_kwargs, "messages": messages}
        followup = self._client.messages.create(**followup_kwargs)

        parts = [b.text for b in followup.content if hasattr(b, "text")]
        return "\n".join(parts)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> Any:
        """
        Dispatch a tool call to the appropriate tool function.
        Subclasses can override to add more tools.
        """
        if tool_name == "web_search":
            from hedge_fund.tools.web_search import web_search
            return web_search(tool_input.get("query", ""))

        if tool_name == "get_price":
            from hedge_fund.tools.market_data import get_current_price
            return get_current_price(tool_input.get("ticker", ""))

        if tool_name == "get_fundamentals":
            from hedge_fund.tools.market_data import get_fundamentals
            return json.dumps(get_fundamentals(tool_input.get("ticker", "")))

        if tool_name == "run_python":
            from hedge_fund.tools.code_executor import execute_python
            return json.dumps(execute_python(tool_input.get("code", "")))

        if tool_name == "get_sec_filing":
            from hedge_fund.tools.sec_filings import get_filing_summary
            return get_filing_summary(
                tool_input.get("ticker", ""),
                tool_input.get("form_type", "10-K"),
            )

        return f"Unknown tool: {tool_name}"

    # ── Structured Output ─────────────────────────────────────────────────────

    def parse_structured_output(
        self,
        response_text: str,
        schema: Type[T],
        prompt_hint: str = "",
    ) -> T:
        """
        Ask Claude to parse its own response into a Pydantic model.
        Falls back to asking for a JSON rewrite if parsing fails.
        """
        # Try to find JSON block in the response
        json_str = self._extract_json(response_text)
        if json_str:
            try:
                return schema.model_validate_json(json_str)
            except Exception:
                pass

        # Ask Claude to reformat as JSON
        reformat_prompt = (
            f"Convert the following text into valid JSON matching this schema:\n"
            f"{schema.model_json_schema()}\n\n"
            f"Text to convert:\n{response_text}\n\n"
            f"Return ONLY the JSON object, no explanation."
        )
        json_response = self._call_llm(reformat_prompt)
        json_str = self._extract_json(json_response) or json_response
        try:
            return schema.model_validate_json(json_str)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Structured parse failed: {e}")
            raise

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first JSON object or array from text."""
        import re
        # Try ```json blocks first
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Try bare JSON object
        match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    # ── Memory Helpers ────────────────────────────────────────────────────────

    def remember(self, text: str, metadata: Optional[dict] = None) -> None:
        """Store something in long-term memory."""
        self.memory.store(text, metadata=metadata)

    def recall(self, query: str, n: int = 3) -> list[dict]:
        """Retrieve relevant memories."""
        return self.memory.search(query, n_results=n)

    # ── Health / Ops ──────────────────────────────────────────────────────────

    def health_status(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "is_healthy": self._error_count < 5,
            "task_queue_depth": self.get_queue_depth(),
            "avg_response_time_s": round(self._last_response_time, 2),
            "error_rate_pct": round(
                self._error_count / max(self._task_count, 1) * 100, 1
            ),
            "last_heartbeat": datetime.utcnow().isoformat(),
        }

    # ── Main entrypoint ───────────────────────────────────────────────────────

    def process_inbox(self) -> list[str]:
        """
        Process all pending messages in the inbox.
        Default implementation: call handle_message for each.
        Returns list of response message_ids.
        """
        messages = self.receive_messages()
        responses = []
        for msg in messages:
            try:
                resp_id = self.handle_message(msg)
                if resp_id:
                    responses.append(resp_id)
            except Exception as e:
                logger.error(f"[{self.agent_id}] Error handling message {msg.message_id}: {e}")
        return responses

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        """
        Override in subclasses to define message-handling logic.
        Return the message_id of any reply sent, or None.
        """
        logger.debug(f"[{self.agent_id}] Unhandled message from {message.sender}: {message.subject}")
        return None

    def run_task(self, task_description: str, context: Optional[str] = None) -> str:
        """
        Run a free-form task and return the LLM response.
        The workhorse method for directive-driven agents.
        """
        logger.info(f"[{self.agent_id}] Running task: {task_description[:80]}")
        response = self._call_llm(task_description, additional_context=context)
        # Store the task and response in long-term memory
        self.memory.store(
            f"TASK: {task_description[:200]}\nRESPONSE: {response[:400]}",
            metadata={"type": "task_log", "timestamp": datetime.utcnow().isoformat()},
        )
        return response
