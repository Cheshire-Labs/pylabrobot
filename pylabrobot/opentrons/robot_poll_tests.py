"""Tests for the wait budgets a command is polled under: what the deadline covers,
what the plunger headroom is allowed to raise, and that a caller can set both.

A deployment whose own dispatcher aborts overrunning commands, or one on a slow
link, has to be able to move these without editing the module. A budget fixed in
here fires before the dispatcher's and takes the decision away from the operator.
"""

import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from pylabrobot.opentrons.flex import OpentronsFlex
from pylabrobot.opentrons.robot import (
  DEFAULT_COMMAND_TIMEOUT,
  DEFAULT_STATUS_POLL_INTERVAL,
  _command_budget,
)
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


def _flex(
  transport: Optional[ChatterboxTransport] = None,
  command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
  status_poll_interval: float = DEFAULT_STATUS_POLL_INTERVAL,
) -> OpentronsFlex:
  robot = OpentronsFlex(
    deck=FlexDeck(),
    host="localhost",
    transport=transport or ChatterboxTransport(),
    command_timeout=command_timeout,
    status_poll_interval=status_poll_interval,
  )
  robot.run_id = "run-1"
  return robot


class CommandBudgetTests(unittest.TestCase):
  """The budget arithmetic on its own, so a regression fails in milliseconds
  instead of sitting out the wait it wrongly imposed."""

  def test_a_command_with_no_plunger_travel_keeps_the_budget_it_was_given(self):
    self.assertEqual(_command_budget(0.01, {}), 0.01)

  def test_a_command_that_names_its_plunger_travel_gets_headroom_on_top(self):
    # 1000 uL at 10 uL/s is 100 s of travel, which no fixed budget covers.
    self.assertEqual(_command_budget(0.01, {"volume": 1000.0, "flowRate": 10.0}), 130.0)

  def test_the_defaults_outlast_a_gripper_labware_move(self):
    self.assertEqual(DEFAULT_COMMAND_TIMEOUT, 120.0)
    self.assertEqual(DEFAULT_STATUS_POLL_INTERVAL, 0.2)

  def test_a_budget_of_zero_or_less_is_refused(self):
    """A zero poll interval spins the robot-server; a zero command budget never polls."""
    with self.assertRaises(ValueError):
      _flex(command_timeout=0.0)
    with self.assertRaises(ValueError):
      _flex(command_timeout=-5.0)
    with self.assertRaises(ValueError):
      _flex(status_poll_interval=0.0)


class CommandPollTests(unittest.IsolatedAsyncioTestCase):
  async def test_a_command_that_finished_during_the_last_sleep_is_not_a_timeout(self):
    # The status read after the deadline is the one that sees it, and aborting a
    # move that already succeeded is worse than one extra read.
    transport = _ScriptedStatusTransport(succeed_after=1)
    robot = _flex(transport)

    result = await robot._execute_command("home", {}, timeout=0.01)

    self.assertEqual(result["status"], "succeeded")

  async def test_a_command_with_no_budget_of_its_own_is_polled_for_the_robot_s(self):
    """Without this the robot falls back to a fixed ceiling nothing above can raise,
    and a home the robot-server takes minutes over is aborted mid-move."""
    robot = _flex(_ScriptedStatusTransport(), command_timeout=0.05, status_poll_interval=0.01)

    with self.assertRaises(RuntimeError) as caught:
      await robot._execute_command("home", {})

    self.assertIn("timed out after 0.05s", str(caught.exception))

  async def test_a_caller_may_widen_one_command_past_the_robot_default(self):
    robot = _flex(_ScriptedStatusTransport(), command_timeout=0.1, status_poll_interval=0.01)

    started = time.monotonic()
    with self.assertRaises(RuntimeError):
      await robot.send_command("home", {}, timeout=0.6)

    self.assertGreater(time.monotonic() - started, 0.5)

  async def test_a_home_is_polled_for_the_robot_s_budget(self):
    """``home`` used to send no budget at all, so it inherited whatever fixed default
    the command path carried rather than the one the caller set."""
    robot = _flex(_ScriptedStatusTransport(), command_timeout=0.05, status_poll_interval=0.01)

    with self.assertRaises(RuntimeError) as caught:
      await robot.home()

    self.assertIn("timed out after 0.05s", str(caught.exception))

  async def test_a_home_may_carry_a_budget_of_its_own(self):
    """A full gantry-and-pipette home outlasts the other commands, so widening it
    must not mean widening every command."""
    robot = _flex(_ScriptedStatusTransport(), command_timeout=0.05, status_poll_interval=0.01)

    with self.assertRaises(RuntimeError) as caught:
      await robot.home(timeout=0.2)

    self.assertIn("timed out after 0.2s", str(caught.exception))

  async def test_the_poll_interval_is_the_delay_between_two_status_reads(self):
    """A deployment on a slow link turns the rate down so it stops hammering the
    robot-server, and a fixed delay in here would ignore it.

    Asserted on the delays asked for rather than on how many reads fit in a wall-clock
    window, which counts how busy the machine is as much as it counts the rate.
    """
    transport = _ScriptedStatusTransport(succeed_after=3)
    robot = _flex(transport, command_timeout=60.0, status_poll_interval=0.05)
    slept: List[float] = []

    async def record(seconds: float) -> None:
      slept.append(seconds)

    with patch("asyncio.sleep", record):
      await robot._execute_command("home", {})

    self.assertEqual(transport.status_reads, 4)
    self.assertEqual(slept, [0.05, 0.05, 0.05])
