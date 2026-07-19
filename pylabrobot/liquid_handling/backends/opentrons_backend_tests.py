import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("ot_api")

from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.liquid_handling.backends.opentrons_backend import (
  _LIQUID_GANGED_FIELDS,
  _OT_DECK_IS_ADDRESSABLE_AREA_VERSION,
  OpentronsOT2Backend,
)
from pylabrobot.liquid_handling.errors import NoChannelError
from pylabrobot.liquid_handling.standard import (
  Drop,
  Mix,
  Pickup,
  SingleChannelAspiration,
)
from pylabrobot.resources import Coordinate, Tip, no_volume_tracking
from pylabrobot.resources.celltreat import celltreat_96_wellplate_350uL_Fb
from pylabrobot.resources.trough import Trough
from pylabrobot.resources.opentrons import (
  OTDeck,
  opentrons_96_filtertiprack_20ul,
  opentrons_96_filtertiprack_200ul,
)
from pylabrobot.resources.well import Well


def _mock_define(lw):
  return {"data": {"definitionUri": f'lw["namespace"]/{lw["metadata"]["displayName"]}/1'}}


def _mock_add(load_name, namespace, ot_location, version, labware_id, display_name):
  return labware_id


def _mock_health_get():
  return {
    "api_version": "7.0.1",
  }


class OpentronsBackendSetupTests(unittest.IsolatedAsyncioTestCase):
  """Tests for setup and stop"""

  @patch("ot_api.runs.create")
  @patch("ot_api.health.home")
  @patch("ot_api.lh.add_mounted_pipettes")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  @patch("ot_api.health.get")
  async def test_setup(
    self,
    mock_health_get,
    mock_define,
    mock_add,
    mock_add_mounted_pipettes,
    mock_home,
    mock_create,
  ):
    mock_create.return_value = "run-id"
    mock_add_mounted_pipettes.return_value = (
      {"pipetteId": "left-pipette-id", "name": "p20_single_gen2"},
      {"pipetteId": "right-pipette-id", "name": "p20_single_gen2"},
    )
    mock_add.side_effect = _mock_add
    mock_define.side_effect = _mock_define
    mock_health_get.side_effect = _mock_health_get

    self.backend = OpentronsOT2Backend(host="localhost", port=1338)
    self.lh = LiquidHandler(backend=self.backend, deck=OTDeck())
    await self.lh.setup()

  def test_serialize(self):
    serialized = OpentronsOT2Backend(host="localhost", port=1337).serialize()
    self.assertEqual(
      serialized,
      {"type": "OpentronsOT2Backend", "host": "localhost", "port": 1337},
    )
    self.assertEqual(
      OpentronsOT2Backend.deserialize(serialized).__class__.__name__,
      "OpentronsOT2Backend",
    )


class OpentronsBackendCommandTests(unittest.IsolatedAsyncioTestCase):
  """Tests Opentrons commands"""

  @patch("ot_api.runs.create")
  @patch("ot_api.health.home")
  @patch("ot_api.lh.add_mounted_pipettes")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  @patch("ot_api.health.get")
  async def asyncSetUp(
    self,
    mock_health_get,
    mock_define,
    mock_add,
    mock_add_mounted_pipettes,
    mock_home,
    mock_create,
  ):
    mock_add.side_effect = _mock_add
    mock_define.side_effect = _mock_define
    mock_add_mounted_pipettes.return_value = (
      {"pipetteId": "left-pipette-id", "name": "p20_single_gen2"},
      {"pipetteId": "right-pipette-id", "name": "p20_single_gen2"},
    )
    mock_create.return_value = "run-id"
    mock_health_get.side_effect = _mock_health_get

    self.backend = OpentronsOT2Backend(host="localhost", port=1338)
    self.deck = OTDeck()
    self.lh = LiquidHandler(backend=self.backend, deck=self.deck)
    await self.lh.setup()

    self.tip_rack = opentrons_96_filtertiprack_20ul(name="tip_rack")
    self.deck.assign_child_at_slot(self.tip_rack, slot=1)
    self.plate = celltreat_96_wellplate_350uL_Fb(name="plate")
    self.deck.assign_child_at_slot(self.plate, slot=11)

  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.define")
  @patch("ot_api.labware.add")
  async def test_tip_pick_up(self, mock_add=None, mock_define=None, mock_pick_up_tip=None):
    assert mock_pick_up_tip is not None and mock_define is not None and mock_add is not None
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add

    def assert_parameters(labware_id, well_name, pipette_id, offset_x, offset_y, offset_z):
      self.assertEqual(labware_id, self.backend.get_ot_name("tip_rack"))
      self.assertEqual(well_name, self.backend.get_ot_name("tip_rack_A1"))
      self.assertEqual(pipette_id, "left-pipette-id")
      self.assertEqual(offset_x, offset_x)
      self.assertEqual(offset_y, offset_y)
      self.assertEqual(offset_z, offset_z)

    mock_pick_up_tip.side_effect = assert_parameters

    await self.lh.pick_up_tips(self.tip_rack["A1"])

  @patch("ot_api.lh.drop_tip")
  async def test_tip_drop(self, mock_drop_tip):
    def assert_parameters(labware_id, well_name, pipette_id, offset_x, offset_y, offset_z):
      self.assertEqual(well_name, self.backend.get_ot_name("tip_rack_A1"))
      self.assertEqual(well_name, self.backend.get_ot_name("tip_rack_A1"))
      self.assertEqual(pipette_id, "left-pipette-id")
      self.assertEqual(offset_x, offset_x)
      self.assertEqual(offset_y, offset_y)
      self.assertEqual(offset_z, offset_z)

    mock_drop_tip.side_effect = assert_parameters

    await self.test_tip_pick_up()
    await self.lh.drop_tips(self.tip_rack["A1"])

  @patch("ot_api.lh.aspirate_in_place")
  @patch("ot_api.lh.move_arm")
  async def test_aspirate(self, mock_move=None, mock_aspirate=None):
    assert mock_aspirate is not None and mock_move is not None

    def assert_parameters(
      volume,
      flow_rate,
      pipette_id,
    ):
      self.assertEqual(pipette_id, "left-pipette-id")
      self.assertEqual(volume, 10)
      self.assertEqual(flow_rate, 3.78)

    mock_aspirate.side_effect = assert_parameters

    await self.test_tip_pick_up()
    self.plate.get_well("A1").tracker.set_volume(10)
    await self.lh.aspirate(self.plate["A1"], vols=[10])

  @patch("ot_api.lh.dispense_in_place")
  @patch("ot_api.lh.move_arm")
  async def test_dispense(self, mock_move, mock_dispense):
    def assert_parameters(
      volume,
      flow_rate,
      pipette_id,
    ):
      self.assertEqual(pipette_id, "left-pipette-id")
      self.assertEqual(volume, 10)
      self.assertEqual(flow_rate, 7.56)

    mock_dispense.side_effect = assert_parameters

    await self.test_aspirate()  # aspirate first
    with no_volume_tracking():
      await self.lh.dispense(self.plate["A1"], vols=[10])

  # -- characterization of the remaining ot_api call sites (Phase 0 safety net) --

  @patch("ot_api.health.home")
  async def test_home_calls_health_home(self, mock_home):
    """home() issues exactly one ot_api.health.home() call."""
    await self.backend.home()
    mock_home.assert_called_once_with()

  @patch("ot_api.modules.list_connected_modules")
  async def test_list_connected_modules_passthrough(self, mock_modules):
    """list_connected_modules() returns ot_api.modules.list_connected_modules() verbatim."""
    mock_modules.return_value = [{"id": "tempdeck"}]
    result = await self.backend.list_connected_modules()
    mock_modules.assert_called_once_with()
    self.assertEqual(result, [{"id": "tempdeck"}])

  @patch("ot_api.run_id", "run-id", create=True)
  @patch("ot_api.requestor.post")
  async def test_stop_cancels_active_run_and_clears_pipettes(self, mock_post):
    """stop() cancels the active run through the requestor and clears mounted pipettes."""
    await self.backend.stop()
    mock_post.assert_called_once_with("/runs/run-id/cancel")
    self.assertIsNone(self.backend.left_pipette)
    self.assertIsNone(self.backend.right_pipette)

  @patch("ot_api.lh.drop_tip_in_place")
  @patch("ot_api.lh.move_to_addressable_area_for_drop_tip")
  @patch("ot_api.lh.drop_tip")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.define")
  @patch("ot_api.labware.add")
  async def test_tip_drop_to_trash_uses_addressable_area(
    self,
    mock_add,
    mock_define,
    mock_pick_up_tip,
    mock_drop_tip,
    mock_to_trash,
    mock_drop_in_place,
  ):
    """At api_version >= 7.1.0 a discard to the deck trash routes via the addressable
    area (move_to_addressable_area_for_drop_tip + drop_tip_in_place), not drop_tip."""
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    self.backend.ot_api_version = _OT_DECK_IS_ADDRESSABLE_AREA_VERSION

    await self.lh.pick_up_tips(self.tip_rack["A1"])
    await self.lh.discard_tips()

    mock_to_trash.assert_called_once()
    mock_drop_in_place.assert_called_once()
    mock_drop_tip.assert_not_called()

  # -- free-space channel motion --

  def _save_position_result(self, x, y, z):
    """The shape savePosition returns: the pipette critical point, in the robot frame."""
    return {"result": {"positionId": "pos-1", "position": {"x": x, "y": y, "z": z}}}

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_x_substitutes_only_x(self, mock_move):
    """move_channel_x reads the live pose, replaces x, and leaves y and z untouched."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ) as mock_command:
      await self.backend.move_channel_x(0, 50.0)

    mock_command.assert_called_once_with("savePosition", {"pipetteId": "left-pipette-id"})
    corner = self.deck.slot_locations[0]
    kwargs = mock_move.call_args.kwargs
    self.assertEqual(kwargs["pipette_id"], "left-pipette-id")
    self.assertAlmostEqual(kwargs["location_x"], 50.0 - corner.x)
    self.assertAlmostEqual(kwargs["location_y"], 2.0)
    self.assertAlmostEqual(kwargs["location_z"], 3.0)

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_y_substitutes_only_y(self, mock_move):
    """move_channel_y reads the live pose, replaces y, and leaves x and z untouched."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ):
      await self.backend.move_channel_y(0, 60.0)

    corner = self.deck.slot_locations[0]
    kwargs = mock_move.call_args.kwargs
    self.assertAlmostEqual(kwargs["location_x"], 1.0)
    self.assertAlmostEqual(kwargs["location_y"], 60.0 - corner.y)
    self.assertAlmostEqual(kwargs["location_z"], 3.0)

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_z_substitutes_only_z(self, mock_move):
    """move_channel_z reads the live pose, replaces z, and leaves x and y untouched."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ):
      await self.backend.move_channel_z(0, 70.0)

    corner = self.deck.slot_locations[0]
    kwargs = mock_move.call_args.kwargs
    self.assertAlmostEqual(kwargs["location_x"], 1.0)
    self.assertAlmostEqual(kwargs["location_y"], 2.0)
    self.assertAlmostEqual(kwargs["location_z"], 70.0 - corner.z)

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_to_applies_every_supplied_axis_in_one_motion(self, mock_move):
    """A combined move applies all supplied axes with a SINGLE pose read and a SINGLE motion,
    rather than the axis-by-axis staircase three separate move_channel_* calls would trace."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ) as mock_command:
      await self.backend.move_channel_to(0, x=50.0, y=60.0, z=70.0)

    mock_command.assert_called_once_with("savePosition", {"pipetteId": "left-pipette-id"})
    self.assertEqual(mock_move.call_count, 1)
    corner = self.deck.slot_locations[0]
    kwargs = mock_move.call_args.kwargs
    self.assertAlmostEqual(kwargs["location_x"], 50.0 - corner.x)
    self.assertAlmostEqual(kwargs["location_y"], 60.0 - corner.y)
    self.assertAlmostEqual(kwargs["location_z"], 70.0 - corner.z)

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_to_holds_the_axes_left_unspecified(self, mock_move):
    """An omitted axis holds its current value, so a z-only combined move does not disturb x or y."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ):
      await self.backend.move_channel_to(0, z=70.0)

    corner = self.deck.slot_locations[0]
    kwargs = mock_move.call_args.kwargs
    self.assertAlmostEqual(kwargs["location_x"], 1.0)
    self.assertAlmostEqual(kwargs["location_y"], 2.0)
    self.assertAlmostEqual(kwargs["location_z"], 70.0 - corner.z)

  async def test_move_channel_to_rejects_a_move_with_no_axes(self):
    """Supplying no axis is a caller mistake, not a silent no-op move to the current pose."""
    with self.assertRaises(ValueError):
      await self.backend.move_channel_to(0)

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_target_is_deck_frame(self, mock_move):
    """The axis target is a deck-frame coordinate, so a move to slot 1's corner lands on the
    robot origin. savePosition reports the robot frame, so mixing the two would silently offset
    every move by the deck origin."""
    corner = self.deck.slot_locations[0]
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(0.0, 0.0, 0.0)
    ):
      await self.backend.move_channel_x(0, corner.x)

    self.assertAlmostEqual(mock_move.call_args.kwargs["location_x"], 0.0)

  @patch("ot_api.lh.move_arm")
  async def test_move_channel_uses_right_pipette_for_channel_1(self, mock_move):
    """Channel 1 is the right mount; the pose query must name that pipette."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ) as mock_command:
      await self.backend.move_channel_x(1, 50.0)

    mock_command.assert_called_once_with("savePosition", {"pipetteId": "right-pipette-id"})
    self.assertEqual(mock_move.call_args.kwargs["pipette_id"], "right-pipette-id")

  async def test_move_channel_rejects_unknown_channel(self):
    """An out-of-range channel raises NoChannelError rather than a masked error."""
    with self.assertRaises(NoChannelError):
      await self.backend.move_channel_x(7, 50.0)

  async def test_get_channel_position_reads_via_save_position(self):
    """get_channel_position queries savePosition for the channel and returns its deck-frame point."""
    with patch.object(
      self.backend, "_run_command", return_value=self._save_position_result(1.0, 2.0, 3.0)
    ) as mock_command:
      position = await self.backend.get_channel_position(1)
    mock_command.assert_called_once_with("savePosition", {"pipetteId": "right-pipette-id"})
    expected = self.backend._robot_to_deck_frame(Coordinate(1.0, 2.0, 3.0))
    self.assertEqual((position.x, position.y, position.z), (expected.x, expected.y, expected.z))


def _make_backend_with_pipettes(left_name="p300_single_gen2", right_name="p20_single_gen2"):
  """Create a backend with pipette state set directly (no ot_api needed)."""
  backend = OpentronsOT2Backend.__new__(OpentronsOT2Backend)
  backend.left_pipette = {"name": left_name, "pipetteId": "left-id"} if left_name else None
  backend.right_pipette = {"name": right_name, "pipetteId": "right-id"} if right_name else None
  backend.left_pipette_has_tip = False
  backend.right_pipette_has_tip = False
  return backend


class OpentronsSharedHelperTests(unittest.TestCase):
  """Tests for _get_pickup_pipette, _get_drop_pipette, _get_liquid_pipette, _set_tip_state."""

  def setUp(self):
    self.backend = _make_backend_with_pipettes()
    self.deck = OTDeck()
    self.tip_rack = opentrons_96_filtertiprack_20ul(name="tip_rack")
    self.deck.assign_child_at_slot(self.tip_rack, slot=1)
    self.tip_spot = self.tip_rack.get_item("A1")
    self.tip_20 = Tip(
      has_filter=True,
      total_tip_length=39.2,
      maximal_volume=20,
      fitting_depth=8.25,
      name="test_tip_20",
    )
    self.tip_300 = Tip(
      has_filter=False,
      total_tip_length=51.0,
      maximal_volume=300,
      fitting_depth=8.0,
      name="test_tip_300",
    )

  # -- _get_pickup_pipette --

  def test_get_pickup_pipette_resolves_the_requested_channel_to_its_pipette(self):
    """The requested channel names the pipette, rather than the backend re-deciding from the tip.

    A channel index is a tip-bearing position, and on a multi-nozzle pipette several of them name
    the same pipette, so the mapping only runs one way. Honoring the caller's channels is also
    what lets a method written for another liquid handler run here unchanged.
    """
    ops_20 = [Pickup(resource=self.tip_spot, offset=Coordinate.zero(), tip=self.tip_20)]
    ops_300 = [Pickup(resource=self.tip_spot, offset=Coordinate.zero(), tip=self.tip_300)]
    self.assertEqual(self.backend._get_pickup_pipette(ops_20, [1]), "right-id")
    self.assertEqual(self.backend._get_pickup_pipette(ops_300, [0]), "left-id")

  def test_get_pickup_pipette_rejects_a_tip_the_requested_pipette_cannot_take(self):
    """Honoring the channel is not the same as accepting an impossible tip.

    LiquidHandler normally avoids this by consulting can_pick_up_tip while choosing channels; this
    is the backstop for a caller that reaches the backend directly.
    """
    ops = [Pickup(resource=self.tip_spot, offset=Coordinate.zero(), tip=self.tip_300)]
    with self.assertRaises(NoChannelError):
      self.backend._get_pickup_pipette(ops, [1])

  def test_get_pickup_pipette_raises_when_tip_already_mounted(self):
    self.backend.right_pipette_has_tip = True
    ops = [Pickup(resource=self.tip_spot, offset=Coordinate.zero(), tip=self.tip_20)]
    with self.assertRaises(NoChannelError):
      self.backend._get_pickup_pipette(ops, [1])

  # -- _deck_to_robot_frame --

  def test_deck_to_robot_frame_maps_slot1_corner_to_robot_origin(self):
    """The deck->robot transform subtracts slot 1's corner, so a deck-frame point at slot 1's
    corner becomes the robot origin and a point offset from it keeps that offset."""
    self.backend.set_deck(self.deck)
    corner = self.deck.slot_locations[0]
    self.assertEqual(self.backend._deck_to_robot_frame(corner), Coordinate(0, 0, 0))
    self.assertEqual(
      self.backend._deck_to_robot_frame(corner + Coordinate(10, 20, 3)),
      Coordinate(10, 20, 3),
    )

  # -- _get_drop_pipette --

  def test_get_drop_pipette_resolves_the_requested_channel_to_its_pipette(self):
    self.backend.right_pipette_has_tip = True
    ops = [Drop(resource=self.tip_spot, offset=Coordinate.zero(), tip=self.tip_20)]
    self.assertEqual(self.backend._get_drop_pipette(ops, [1]), "right-id")

  def test_get_drop_pipette_raises_when_no_tip(self):
    ops = [Drop(resource=self.tip_spot, offset=Coordinate.zero(), tip=self.tip_20)]
    with self.assertRaises(NoChannelError):
      self.backend._get_drop_pipette(ops, [1])

  # -- _get_liquid_pipette --

  def test_get_liquid_pipette_resolves_the_requested_channel_to_its_pipette(self):
    """Which pipette aspirates follows from the channel asked for, not from the volume."""
    self.backend.left_pipette_has_tip = True
    self.backend.right_pipette_has_tip = True
    well = Well(name="w", size_x=5, size_y=5, size_z=10, max_volume=350)
    ops = [
      SingleChannelAspiration(
        resource=well,
        offset=Coordinate.zero(),
        tip=self.tip_300,
        volume=100,
        flow_rate=None,
        liquid_height=None,
        blow_out_air_volume=None,
        mix=None,
      )
    ]
    self.assertEqual(self.backend._get_liquid_pipette(ops, [0]), "left-id")
    self.assertEqual(self.backend._get_liquid_pipette(ops, [1]), "right-id")

  def test_get_liquid_pipette_raises_without_tip(self):
    well = Well(name="w", size_x=5, size_y=5, size_z=10, max_volume=350)
    ops = [
      SingleChannelAspiration(
        resource=well,
        offset=Coordinate.zero(),
        tip=self.tip_20,
        volume=5,
        flow_rate=None,
        liquid_height=None,
        blow_out_air_volume=None,
        mix=None,
      )
    ]
    with self.assertRaises(NoChannelError):
      self.backend._get_liquid_pipette(ops, [1])

  # -- _set_tip_state --

  def test_set_tip_state_left(self):
    self.backend._set_tip_state("left-id", True)
    self.assertTrue(self.backend.left_pipette_has_tip)
    self.assertFalse(self.backend.right_pipette_has_tip)

  def test_set_tip_state_right(self):
    self.backend._set_tip_state("right-id", True)
    self.assertFalse(self.backend.left_pipette_has_tip)
    self.assertTrue(self.backend.right_pipette_has_tip)


class OpentronsMultiChannelTests(unittest.TestCase):
  """An 8-channel pipette is 8 nozzles driven by ONE plunger.

  Those nozzles are 8 tip-bearing positions, so they are 8 pylabrobot channels, but they share a
  single volume. These pin both halves: the channel count LiquidHandler needs before it will
  dispatch an 8-channel method at all, and the ganged-volume constraint that keeps the shared
  plunger honest.
  """

  def setUp(self):
    self.backend = _make_backend_with_pipettes(left_name="p300_multi_gen2", right_name=None)
    self.deck = OTDeck()
    self.backend.set_deck(self.deck)
    self.tip_rack = opentrons_96_filtertiprack_200ul(name="tip_rack")
    self.deck.assign_child_at_slot(self.tip_rack, slot=1)
    self.tip_200 = Tip(
      has_filter=True,
      total_tip_length=51.0,
      maximal_volume=200,
      fitting_depth=8.0,
      name="test_tip_200",
    )
    self.plate = celltreat_96_wellplate_350uL_Fb(name="geom_plate")
    self.deck.assign_child_at_slot(self.plate, slot=11)
    self.trough = Trough(
      name="geom_trough",
      size_x=107.0,
      size_y=85.0,
      size_z=44.0,
      max_volume=100_000,
      material_z_thickness=1.0,
    )
    self.deck.assign_child_at_slot(self.trough, slot=2)

  def _trough_ops(self, resource, count: int, back_offset: float):
    """Ops that all name one resource, spread back-to-front the way LiquidHandler builds them."""
    return [
      Pickup(
        resource=resource,
        offset=Coordinate(0, back_offset - i * 9.0, 0),
        tip=self.tip_200,
      )
      for i in range(count)
    ]

  def _aspirations(self, volumes):
    well = Well(name="w", size_x=5, size_y=5, size_z=10, max_volume=350)
    return [
      SingleChannelAspiration(
        resource=well,
        offset=Coordinate.zero(),
        tip=self.tip_200,
        volume=volume,
        flow_rate=None,
        liquid_height=None,
        blow_out_air_volume=None,
        mix=None,
      )
      for volume in volumes
    ]

  def test_reports_one_channel_per_nozzle(self):
    """Reporting the mount count instead would make LiquidHandler reject an 8-channel method
    before it ever reached the robot, which is what portability across liquid handlers rests on."""
    self.assertEqual(self.backend.num_channels, 8)

  def test_channel_groups_report_the_shared_plunger(self):
    """num_channels counts tip positions; this is what says how many distinct volumes the head can
    deliver, which is what a caller planning per-channel volumes actually needs."""
    self.assertEqual(self.backend.channel_groups, [list(range(8))])

  def test_a_channel_names_its_pipette(self):
    self.assertEqual(self.backend.pipette_name_for_channel(5), "p300_multi_gen2")

  def test_two_single_pipettes_are_two_independent_groups(self):
    """Two mounts each with their own plunger CAN deliver two different volumes at once, which is
    the case the single ganged group must not be confused with."""
    backend = _make_backend_with_pipettes(
      left_name="p300_single_gen2", right_name="p20_single_gen2"
    )
    self.assertEqual(backend.channel_groups, [[0], [1]])

  def test_a_multi_and_a_single_gang_only_within_their_own_pipette(self):
    backend = _make_backend_with_pipettes(left_name="p300_multi_gen2", right_name="p20_single_gen2")
    self.assertEqual(backend.channel_groups, [list(range(8)), [8]])

  def test_every_nozzle_resolves_to_the_same_pipette(self):
    self.assertEqual({self.backend._pipette_id_for_channel(c) for c in range(8)}, {"left-id"})

  def test_channel_beyond_the_nozzle_count_is_rejected(self):
    with self.assertRaises(NoChannelError):
      self.backend._pipette_id_for_channel(8)

  def test_can_pick_up_tip_answers_for_every_nozzle(self):
    """can_pick_up_tip is how LiquidHandler picks channels, so it has to answer for all 8."""
    self.assertTrue(all(self.backend.can_pick_up_tip(c, self.tip_200) for c in range(8)))
    self.assertFalse(self.backend.can_pick_up_tip(8, self.tip_200))

  def test_partial_nozzle_request_is_refused(self):
    """The robot runs its full configuration regardless, engaging all 8 nozzles while the caller
    believed it addressed 3. Refusing is honest until configureNozzleLayout is wired up."""
    ops = [
      Pickup(resource=self.tip_rack.get_item(i), offset=Coordinate.zero(), tip=self.tip_200)
      for i in range(3)
    ]
    with self.assertRaises(NoChannelError):
      self.backend._get_pickup_pipette(ops, [0, 1, 2])

  def test_uniform_volumes_collapse_to_the_single_plunger_volume(self):
    self.assertEqual(self.backend._ganged_value(self._aspirations([100.0] * 8), "volume"), 100.0)

  def test_differing_volumes_are_refused(self):
    """One plunger cannot deliver 8 different volumes. This is the real hardware boundary a
    portable method meets: uniform volumes run anywhere, per-channel volumes need 8 plungers."""
    with self.assertRaises(ValueError):
      self.backend._ganged_value(
        self._aspirations([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]), "volume"
      )

  def test_every_ganged_parameter_is_checked_not_just_volume(self):
    """A shared plunger and a rigid mount fix the volume, flow rate, mix, liquid height, blow-out
    and offset alike. Reading one op and discarding the rest would act on parameters nobody asked
    for, so each is exercised, not just volume."""
    odd_mix = Mix(volume=5.0, repetitions=1, flow_rate=None)
    for attribute, values in (
      ("volume", [100.0] * 7 + [10.0]),
      ("flow_rate", [50.0] * 7 + [10.0]),
      ("liquid_height", [1.0] * 7 + [5.0]),
      ("blow_out_air_volume", [0.0] * 7 + [3.0]),
      ("offset", [Coordinate.zero()] * 7 + [Coordinate(0, 0, 2.0)]),
      ("mix", [None] * 7 + [odd_mix]),
    ):
      with self.subTest(attribute=attribute):
        ops = [
          SingleChannelAspiration(
            resource=Well(name="w", size_x=5, size_y=5, size_z=10, max_volume=350),
            offset=value if attribute == "offset" else Coordinate.zero(),
            tip=self.tip_200,
            volume=value if attribute == "volume" else 100.0,
            flow_rate=value if attribute == "flow_rate" else None,
            liquid_height=value if attribute == "liquid_height" else None,
            blow_out_air_volume=value if attribute == "blow_out_air_volume" else None,
            mix=value if attribute == "mix" else None,
          )
          for value in values
        ]
        with self.assertRaises(ValueError):
          self.backend._require_ganged_parameters(ops, _LIQUID_GANGED_FIELDS)

  def test_targets_spread_across_x_are_refused(self):
    """The nozzles share one x, so a ROW of wells is unreachable however the command is phrased."""
    row = [
      Pickup(resource=self.tip_rack.get_item(f"A{col}"), offset=Coordinate.zero(), tip=self.tip_200)
      for col in range(1, 9)
    ]
    with self.assertRaisesRegex(ValueError, "single column"):
      self.backend._require_nozzle_geometry(row)

  def test_targets_in_a_column_but_off_the_pitch_are_refused(self):
    """One column, right shape, wrong spacing: every other well is 18 mm apart and the rigid array
    cannot stretch to it. This is the pitch rule rather than the column rule."""
    skipped = [
      Pickup(resource=self.tip_rack.get_item(f"{row}1"), offset=Coordinate.zero(), tip=self.tip_200)
      for row in "ACEG"
    ]
    with self.assertRaisesRegex(ValueError, "nozzle pitch"):
      self.backend._require_nozzle_geometry(skipped)

  def test_targets_on_the_nozzle_pitch_are_accepted(self):
    """A real column of distinct wells at the nozzle pitch is exactly what the array reaches, so it
    must pass where the off-pitch and cross-x cases raise."""
    column = [
      Pickup(resource=self.tip_rack.get_item(f"{row}1"), offset=Coordinate.zero(), tip=self.tip_200)
      for row in "ABCDEFGH"
    ]
    self.backend._require_nozzle_geometry(column)  # does not raise

  def test_the_command_names_the_back_most_well(self):
    """A full 8-nozzle configuration is commanded by naming ONE well: the one under the A1 nozzle,
    at the back of the column. Naming any other would offset the whole head by a row."""
    ops = [
      Pickup(resource=self.tip_rack.get_item(f"{row}1"), offset=Coordinate.zero(), tip=self.tip_200)
      for row in "ABCDEFGH"
    ]
    self.assertIs(self.backend._reference_op(ops).resource, self.tip_rack.get_item("A1"))

  def test_a_shared_target_names_the_back_most_nozzle(self):
    """When every nozzle goes into one resource the ops differ only by their spread offsets, so the
    reference nozzle has to be read from those rather than from the resource they all share."""
    spot = self.tip_rack.get_item("A1")
    ops = [
      Pickup(resource=spot, offset=Coordinate(0, y, 0), tip=self.tip_200)
      for y in (-13.5, 31.5, 4.5, -22.5)
    ]
    self.assertEqual(self.backend._reference_op(ops).offset.y, 31.5)

  def test_a_shared_target_takes_the_whole_array_if_it_holds_it(self):
    """A trough is one cavity taking every nozzle, so the caller's spread need not match the pitch;
    what matters is that the array lands inside it."""
    ops = self._trough_ops(self.trough, count=8, back_offset=31.5)
    self.backend._require_nozzle_geometry(ops)

  def test_a_shared_target_too_small_for_the_array_is_refused(self):
    """Eight nozzles aimed at ONE well span eight wells of the plate. Accepting it would draw from
    seven wells nobody named while the liquid tracker debits all of it from the one that was."""
    well = self.plate.get_well("A1")
    ops = self._trough_ops(well, count=8, back_offset=0.0)
    with self.assertRaisesRegex(ValueError, "does not fit"):
      self.backend._require_nozzle_geometry(ops)

  def test_an_array_hanging_off_the_back_of_a_trough_is_refused(self):
    """Fitting is about where the array is anchored, not only how deep the resource is."""
    ops = self._trough_ops(self.trough, count=8, back_offset=self.trough.get_size_y())
    with self.assertRaisesRegex(ValueError, "does not fit"):
      self.backend._require_nozzle_geometry(ops)


class OpentronsMultiChannelCommandTests(unittest.IsolatedAsyncioTestCase):
  """End to end: an 8-channel request reaches the robot as ONE command, not eight.

  One plunger and one rigid nozzle array mean one pickUpTip and one aspirate. Emitting eight would
  describe eight separate physical operations, which is not what the hardware performs.
  """

  @patch("ot_api.runs.create")
  @patch("ot_api.health.home")
  @patch("ot_api.lh.add_mounted_pipettes")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  @patch("ot_api.health.get")
  async def asyncSetUp(
    self,
    mock_health_get,
    mock_define,
    mock_add,
    mock_add_mounted_pipettes,
    mock_home,
    mock_create,
  ):
    mock_add.side_effect = _mock_add
    mock_define.side_effect = _mock_define
    mock_add_mounted_pipettes.return_value = (
      {"pipetteId": "left-pipette-id", "name": "p300_multi_gen2"},
      None,
    )
    mock_create.return_value = "run-id"
    mock_health_get.side_effect = _mock_health_get

    self.backend = OpentronsOT2Backend(host="localhost", port=1338)
    self.deck = OTDeck()
    self.lh = LiquidHandler(backend=self.backend, deck=self.deck)
    await self.lh.setup()

    self.tip_rack = opentrons_96_filtertiprack_200ul(name="tip_rack")
    self.deck.assign_child_at_slot(self.tip_rack, slot=1)
    self.plate = celltreat_96_wellplate_350uL_Fb(name="plate")
    self.deck.assign_child_at_slot(self.plate, slot=11)
    self.column = [self.tip_rack.get_item(f"{row}1") for row in "ABCDEFGH"]
    self.wells = [self.plate.get_well(f"{row}1") for row in "ABCDEFGH"]

  async def test_the_frontend_sizes_its_head_to_the_nozzles(self):
    """Without this, LiquidHandler refuses an 8-channel request before the backend is reached."""
    self.assertEqual(self.backend.num_channels, 8)
    self.assertEqual(len(self.lh.head), 8)

  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_eight_tip_pickup_sends_one_command_naming_the_back_well(
    self, mock_define, mock_add, mock_pick_up
  ):
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    await self.lh.pick_up_tips(self.column)
    mock_pick_up.assert_called_once()
    self.assertEqual(
      mock_pick_up.call_args.kwargs["well_name"], self.backend.get_ot_name("tip_rack_A1")
    )

  @patch("ot_api.lh.aspirate_in_place")
  @patch("ot_api.lh.move_arm")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_eight_channel_aspirate_sends_one_command(
    self, mock_define, mock_add, mock_pick_up, mock_move, mock_aspirate
  ):
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    await self.lh.pick_up_tips(self.column)
    for well in self.wells:
      well.tracker.set_volume(50)
    await self.lh.aspirate(self.wells, vols=[50] * 8)
    mock_aspirate.assert_called_once()
    self.assertEqual(mock_aspirate.call_args.kwargs["volume"], 50)

  @patch("ot_api.lh.move_arm")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_per_channel_volumes_are_refused(
    self, mock_define, mock_add, mock_pick_up, mock_move
  ):
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    """The portability boundary. Uniform volumes run on any liquid handler; genuinely independent
    per-channel volumes need one plunger per channel, which this pipette does not have."""
    await self.lh.pick_up_tips(self.column)
    for well in self.wells:
      well.tracker.set_volume(100)
    with self.assertRaises(ValueError):
      await self.lh.aspirate(self.wells, vols=[10, 20, 30, 40, 50, 60, 70, 80])


class OpentronsSharedTargetTests(unittest.IsolatedAsyncioTestCase):
  """Every nozzle into ONE resource: a trough, or the trash.

  This is the operation a multi-channel pipette exists for, and it is shaped differently from a
  column of wells. LiquidHandler describes it as one resource repeated per channel, carrying the
  nozzle spread in the per-op offsets, so the column-geometry and identical-offset rules that
  govern distinct targets do not apply to it.
  """

  @patch("ot_api.runs.create")
  @patch("ot_api.health.home")
  @patch("ot_api.lh.add_mounted_pipettes")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  @patch("ot_api.health.get")
  async def asyncSetUp(
    self,
    mock_health_get,
    mock_define,
    mock_add,
    mock_add_mounted_pipettes,
    mock_home,
    mock_create,
  ):
    mock_add.side_effect = _mock_add
    mock_define.side_effect = _mock_define
    mock_add_mounted_pipettes.return_value = (
      {"pipetteId": "left-pipette-id", "name": "p300_multi_gen2"},
      None,
    )
    mock_create.return_value = "run-id"
    mock_health_get.return_value = {"api_version": _OT_DECK_IS_ADDRESSABLE_AREA_VERSION}

    self.backend = OpentronsOT2Backend(host="localhost", port=1338)
    self.deck = OTDeck()
    self.lh = LiquidHandler(backend=self.backend, deck=self.deck)
    await self.lh.setup()

    self.tip_rack = opentrons_96_filtertiprack_200ul(name="tip_rack")
    self.deck.assign_child_at_slot(self.tip_rack, slot=1)
    self.column = [self.tip_rack.get_item(f"{row}1") for row in "ABCDEFGH"]
    self.trough = Trough(
      name="trough",
      size_x=107.0,
      size_y=85.0,
      size_z=44.0,
      max_volume=100_000,
      material_z_thickness=1.0,
    )
    self.deck.assign_child_at_slot(self.trough, slot=2)

  @patch("ot_api.lh.aspirate_in_place")
  @patch("ot_api.lh.move_arm")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_eight_nozzles_aspirate_from_one_trough(
    self, mock_define, mock_add, mock_pick_up, mock_move, mock_aspirate
  ):
    """The canonical multi-channel operation. The nozzles are already at a fixed pitch, so a
    single cavity is reachable by all of them at once."""
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    await self.lh.pick_up_tips(self.column)
    self.trough.tracker.set_volume(50_000)
    await self.lh.aspirate([self.trough] * 8, vols=[50] * 8)
    mock_aspirate.assert_called_once()
    self.assertEqual(mock_aspirate.call_args.kwargs["volume"], 50)

  @patch("ot_api.lh.dispense_in_place")
  @patch("ot_api.lh.aspirate_in_place")
  @patch("ot_api.lh.move_arm")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_eight_nozzles_dispense_into_one_trough(
    self, mock_define, mock_add, mock_pick_up, mock_move, mock_aspirate, mock_dispense
  ):
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    await self.lh.pick_up_tips(self.column)
    self.trough.tracker.set_volume(50_000)
    await self.lh.aspirate([self.trough] * 8, vols=[50] * 8)
    await self.lh.dispense([self.trough] * 8, vols=[50] * 8)
    mock_dispense.assert_called_once()
    self.assertEqual(mock_dispense.call_args.kwargs["volume"], 50)

  @patch("ot_api.lh.drop_tip_in_place")
  @patch("ot_api.lh.move_to_addressable_area_for_drop_tip")
  @patch("ot_api.lh.move_arm")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_eight_tips_discard_to_the_trash(
    self, mock_define, mock_add, mock_pick_up, mock_move, mock_trash_drop, mock_drop_in_place
  ):
    """discard_tips spreads the channels across the trash, so its ops carry differing offsets the
    way a trough's do."""
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    await self.lh.pick_up_tips(self.column)
    await self.lh.discard_tips()
    mock_trash_drop.assert_called_once()

  @patch("ot_api.lh.move_arm")
  @patch("ot_api.lh.pick_up_tip")
  @patch("ot_api.labware.add")
  @patch("ot_api.labware.define")
  async def test_a_shared_target_still_refuses_per_channel_volumes(
    self, mock_define, mock_add, mock_pick_up, mock_move
  ):
    """Sharing a target relaxes the position rules, never the single-plunger rule."""
    mock_define.side_effect = _mock_define
    mock_add.side_effect = _mock_add
    await self.lh.pick_up_tips(self.column)
    self.trough.tracker.set_volume(50_000)
    with self.assertRaisesRegex(ValueError, "volume"):
      await self.lh.aspirate([self.trough] * 8, vols=[10, 20, 30, 40, 50, 60, 70, 80])
