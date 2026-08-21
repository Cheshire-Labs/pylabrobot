"""Tests for session liveness and hardware-truth tip reconciliation.

An operator recovering the robot at the instrument stops our run (the
touchscreen only works while no run is current), which invalidates every
run-scoped identity and can leave the driver's per-channel tip bookkeeping
describing tips that are no longer there. These tests pin the three answers:
a typed error when the run died under us, a liveness read that tells the two
apart from a wire failure, and a per-mount reconcile that trusts the hardware
sensor over the in-memory model.
"""

import asyncio
import unittest
from typing import Tuple

from pylabrobot.opentrons.flex import OpentronsFlex
from pylabrobot.opentrons.flex_head import (
  RECONCILE_CLEARED_LOST_TIPS,
  RECONCILE_IN_SYNC,
  RECONCILE_UNTRACKED_TIP,
  FlexHead8,
)
from pylabrobot.opentrons.robot import OpentronsRunNotCurrentError
from pylabrobot.opentrons.transport import ChatterboxTransport
from pylabrobot.resources import set_tip_tracking, set_volume_tracking
from pylabrobot.resources.opentrons.flex_deck import FlexDeck
from pylabrobot.resources.opentrons.flex_tip_racks import flex_96_tiprack_50ul


def _flex_head8() -> Tuple[OpentronsFlex, ChatterboxTransport, FlexHead8]:
  transport = ChatterboxTransport(pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")])
  flex = OpentronsFlex(deck=FlexDeck(), host="localhost", transport=transport)
  asyncio.run(flex.setup())
  head = flex.left
  assert isinstance(head, FlexHead8)
  return flex, transport, head


class TestRunLiveness(unittest.TestCase):
  """run_is_current tells 'the operator took the robot' apart from 'the wire broke'."""

  def test_our_open_run_reads_current(self):
    flex, _, _ = _flex_head8()
    try:
      self.assertTrue(asyncio.run(flex.run_is_current()))
    finally:
      asyncio.run(flex.disconnect())

  def test_no_run_reads_not_current_without_a_wire_read(self):
    transport = ChatterboxTransport(pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")])
    flex = OpentronsFlex(deck=FlexDeck(), host="localhost", transport=transport)
    self.assertFalse(asyncio.run(flex.run_is_current()))

  def test_an_externally_ended_run_reads_not_current(self):
    flex, transport, _ = _flex_head8()
    try:
      transport.end_run_externally()
      self.assertFalse(asyncio.run(flex.run_is_current()))
    finally:
      asyncio.run(flex.disconnect())

  def test_a_command_against_a_dead_run_raises_the_typed_error(self):
    """The robot-server refuses with a bare HTTP error; the driver types it so
    a caller can react with a session rebuild instead of a blind retry."""
    flex, transport, head = _flex_head8()
    try:
      transport.end_run_externally()
      with self.assertRaises(OpentronsRunNotCurrentError):
        asyncio.run(head.get_tip_presence())
    finally:
      asyncio.run(flex.disconnect())

  def test_a_wire_failure_with_a_live_run_keeps_its_own_error(self):
    """Only a dead run earns the typed error; an unreachable or refusing robot
    must not be misread as an operator takeover."""

    class _RefusingTransport(ChatterboxTransport):
      async def post(self, path, json=None):
        if path.endswith("/commands"):
          raise RuntimeError("503: robot busy")
        return await super().post(path, json)

    transport = _RefusingTransport(pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")])
    flex = OpentronsFlex(deck=FlexDeck(), host="localhost", transport=transport)
    asyncio.run(flex.connect())
    asyncio.run(flex.create_run())
    try:
      with self.assertRaises(RuntimeError) as ctx:
        asyncio.run(flex.set_rail_lights(True))
      self.assertNotIsInstance(ctx.exception, OpentronsRunNotCurrentError)
      self.assertIn("503", str(ctx.exception))
    finally:
      asyncio.run(flex.disconnect())

  def test_a_fresh_run_after_an_external_end_accepts_commands_again(self):
    """create_run + initialize is the whole session rebuild: no homing, and the
    run-scoped labware/pipette identities come back fresh."""
    flex, transport, _ = _flex_head8()
    try:
      transport.end_run_externally()
      asyncio.run(flex.create_run())
      asyncio.run(flex.initialize())
      homes = [c for c in transport.commands if c["commandType"] == "home"]
      asyncio.run(flex.set_rail_lights(True))
      self.assertEqual(len(homes), 1)  # only setup()'s own home; the rebuild adds none
    finally:
      asyncio.run(flex.disconnect())


class TestTipReconcile(unittest.TestCase):
  """The sensor is one bool per mount: clearing on absent is the only safe
  automatic repair, and present-while-untracked is reported, never guessed."""

  def setUp(self):
    set_tip_tracking(True)
    set_volume_tracking(True)

  def tearDown(self):
    set_tip_tracking(False)
    set_volume_tracking(False)

  def test_agreeing_states_reconcile_to_in_sync(self):
    flex, _, head = _flex_head8()
    try:
      self.assertEqual(asyncio.run(head.reconcile_tips_with_hardware()), RECONCILE_IN_SYNC)
    finally:
      asyncio.run(flex.stop())

  def test_sensor_absent_while_holding_clears_every_channel(self):
    """The incident's shape: tips dropped at the instrument, model still full."""
    flex, transport, head = _flex_head8()
    try:
      rack = flex_96_tiprack_50ul(name="rack")
      flex.deck.assign_child_at_slot(rack, "C1")
      asyncio.run(head.pick_up_tips(rack, column=0))
      transport.set_tip_detected("left", False)  # the operator dropped them

      outcome = asyncio.run(head.reconcile_tips_with_hardware())

      self.assertEqual(outcome, RECONCILE_CLEARED_LOST_TIPS)
      self.assertTrue(all(t is None for t in head.get_mounted_tips()))
    finally:
      asyncio.run(flex.stop())

  def test_sensor_present_while_empty_reports_and_changes_nothing(self):
    flex, transport, head = _flex_head8()
    try:
      transport.set_tip_detected("left", True)  # a tip only the hardware knows about

      outcome = asyncio.run(head.reconcile_tips_with_hardware())

      self.assertEqual(outcome, RECONCILE_UNTRACKED_TIP)
      self.assertTrue(all(t is None for t in head.get_mounted_tips()))
    finally:
      transport.set_tip_detected("left", False)
      asyncio.run(flex.stop())


class TestUnsafeDiscardTips(unittest.TestCase):
  """The escape for a tip the run does not know about: sensor-trusting travel
  padding plus the checks-skipping drop."""

  def test_discards_via_padded_travel_and_unsafe_drop(self):
    flex, transport, head = _flex_head8()
    try:
      trash = flex.deck.get_trash_area()
      plain_height = head._traversal_height()

      asyncio.run(head.unsafe_discard_tips(trash))

      move_cmd = next(
        c for c in transport.commands if c["commandType"] == "moveToAddressableAreaForDropTip"
      )
      self.assertGreater(move_cmd["params"]["minimumZHeight"], plain_height)
      self.assertIn("unsafe/dropTipInPlace", [c["commandType"] for c in transport.commands])
      self.assertTrue(all(t is None for t in head.get_mounted_tips()))
    finally:
      asyncio.run(flex.stop())


if __name__ == "__main__":
  unittest.main()
