"""Tests for the robot's wait budgets: the transport timeout, the per-command
poll ceiling and the poll interval, and that a caller can set all three.

A deployment on a slow link, or one whose dispatcher aborts overrunning
commands itself, has to be able to move these without editing the module.
"""

import time
import unittest
from typing import Any, Dict, Optional

from pylabrobot.opentrons.flex import OpentronsFlex
from pylabrobot.opentrons.transport import ChatterboxTransport, HttpxTransport
from pylabrobot.resources.opentrons.flex_deck import FlexDeck


class _NeverFinishingTransport(ChatterboxTransport):
  """Answers every command "running", so a poll loop only ends at its deadline.

  Counts the status reads it served, so a test can tell one poll rate from another.
  """

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.status_reads = 0

  async def get(self, path: str) -> Dict[str, Any]:
    if "/commands/" in path:
      self.status_reads += 1
      return {"data": {"id": path.rsplit("/", 1)[-1], "commandType": "", "status": "running"}}
    return await super().get(path)


def _flex(transport: Optional[ChatterboxTransport] = None, **kwargs: float) -> OpentronsFlex:
  return OpentronsFlex(
    deck=FlexDeck(),
    host="localhost",
    transport=transport or ChatterboxTransport(),
    **kwargs,
  )


class RobotTimeoutTests(unittest.IsolatedAsyncioTestCase):
  def test_the_transport_the_robot_builds_carries_the_request_timeout(self):
    robot = OpentronsFlex(deck=FlexDeck(), host="localhost", timeout=7.5)

    transport = robot._transport
    assert isinstance(transport, HttpxTransport)
    self.assertEqual(transport.io.serialize()["timeout"], 7.5)
    self.assertEqual(robot.request_timeout, 7.5)

  def test_an_injected_transport_keeps_its_own_timeout(self):
    """An offline transport is passed in whole; the robot must not restate its budget."""
    transport = ChatterboxTransport()

    robot = _flex(transport, timeout=7.5)

    self.assertIs(robot._transport, transport)

  def test_defaults_match_what_the_robot_shipped_with(self):
    robot = _flex()

    self.assertEqual(robot.request_timeout, 30.0)
    self.assertEqual(robot.command_timeout, 30.0)
    self.assertEqual(robot.status_poll_interval, 0.2)

  async def test_a_command_that_never_finishes_gives_up_at_the_command_timeout(self):
    robot = _flex(_NeverFinishingTransport(), command_timeout=0.3, status_poll_interval=0.01)
    await robot.create_run()

    started = time.monotonic()
    with self.assertRaises(RuntimeError):
      await robot.send_command("home", {})

    self.assertLess(time.monotonic() - started, 5.0)

  async def test_a_caller_may_widen_one_command_past_the_robot_default(self):
    robot = _flex(_NeverFinishingTransport(), command_timeout=0.1, status_poll_interval=0.01)
    await robot.create_run()

    started = time.monotonic()
    with self.assertRaises(RuntimeError):
      await robot.send_command("home", {}, timeout=0.6)

    self.assertGreater(time.monotonic() - started, 0.5)

  async def test_the_poll_interval_paces_the_status_reads(self):
    slow_transport = _NeverFinishingTransport()
    slow = _flex(slow_transport, command_timeout=0.4, status_poll_interval=0.2)
    await slow.create_run()
    with self.assertRaises(RuntimeError):
      await slow.send_command("home", {})

    quick_transport = _NeverFinishingTransport()
    quick = _flex(quick_transport, command_timeout=0.4, status_poll_interval=0.01)
    await quick.create_run()
    with self.assertRaises(RuntimeError):
      await quick.send_command("home", {})

    self.assertGreater(quick_transport.status_reads, slow_transport.status_reads)
