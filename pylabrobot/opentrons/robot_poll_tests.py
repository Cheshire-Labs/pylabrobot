"""Tests for how a command's completion is polled: what the deadline covers, and
what the plunger headroom is allowed to raise."""

import unittest
from typing import Any, Dict, Optional

from pylabrobot.opentrons.flex import OpentronsFlex
from pylabrobot.opentrons.transport import ChatterboxTransport
from pylabrobot.resources.opentrons.flex_deck import FlexDeck


class _ScriptedStatusTransport(ChatterboxTransport):
  """Answers command status from a script instead of succeeding on the first read.

  Counts the reads it served, and reports "running" until ``succeed_after`` reads
  have gone by (forever if None).
  """

  def __init__(self, succeed_after: Optional[int] = None) -> None:
    super().__init__()
    self.status_reads = 0
    self._succeed_after = succeed_after

  async def get(self, path: str) -> Dict[str, Any]:
    if "/commands/" not in path:
      return await super().get(path)
    self.status_reads += 1
    finished = self._succeed_after is not None and self.status_reads > self._succeed_after
    return {
      "data": {
        "id": path.rsplit("/", 1)[-1],
        "commandType": "",
        "status": "succeeded" if finished else "running",
        "result": {},
      }
    }


def _flex(transport: ChatterboxTransport) -> OpentronsFlex:
  return OpentronsFlex(deck=FlexDeck(), host="localhost", transport=transport)


class CommandPollTests(unittest.IsolatedAsyncioTestCase):
  async def test_a_command_that_finished_during_the_last_sleep_is_not_a_timeout(self):
    # The status read after the deadline is the one that sees it, and aborting a
    # move that already succeeded is worse than one extra read.
    transport = _ScriptedStatusTransport(succeed_after=1)
    robot = _flex(transport)
    robot.run_id = "run-1"

    result = await robot._execute_command("home", {}, timeout=0.01)

    self.assertEqual(result["status"], "succeeded")

  async def test_a_command_that_never_finishes_still_times_out(self):
    transport = _ScriptedStatusTransport()
    robot = _flex(transport)
    robot.run_id = "run-1"

    with self.assertRaises(RuntimeError):
      await robot._execute_command("home", {}, timeout=0.01)

  async def test_the_plunger_headroom_is_not_a_floor_under_a_command_with_no_motion(self):
    # A command that names no plunger travel must be allowed a budget below the
    # headroom, or no caller could ever set a short one.
    transport = _ScriptedStatusTransport()
    robot = _flex(transport)
    robot.run_id = "run-1"

    with self.assertRaises(RuntimeError):
      await robot._execute_command("home", {}, timeout=0.01)

    self.assertLess(transport.status_reads, 10, "the budget was raised to the headroom")
