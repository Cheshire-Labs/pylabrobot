import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("ot_api")

from pylabrobot.liquid_handling.backends import OpentronsFlexBackend
from pylabrobot.liquid_handling.standard import (
  GripDirection,
  ResourceDrop,
  ResourceMove,
  ResourcePickup,
  SingleChannelAspiration,
  SingleChannelDispense,
)
from pylabrobot.resources import Coordinate, Resource, Tip
from pylabrobot.resources.opentrons import FlexDeck
from pylabrobot.resources.rotation import Rotation


def _flex_backend() -> OpentronsFlexBackend:
  """A backend with the mounts the sim reports, built without touching ot_api."""
  backend = OpentronsFlexBackend.__new__(OpentronsFlexBackend)
  backend.host, backend.port = "localhost", 31950
  backend._plr_name_to_load_name = {}
  backend._tip_racks = {}
  backend._loaded_labware = {}
  backend._pending_pickup = None
  backend.left_pipette = {"pipetteId": "L", "name": "p50_single_flex"}
  backend.right_pipette = {"pipetteId": "R", "name": "p1000_single_flex"}
  backend.left_pipette_has_tip = backend.right_pipette_has_tip = False
  backend.ot_api_version = "8.0.0"
  backend.traversal_height = 120
  return backend


def _tip(volume: float) -> Tip:
  return Tip(
    name="tip", has_filter=False, total_tip_length=95.0, maximal_volume=volume, fitting_depth=8.0
  )


class FlexBackendUnitTests(unittest.TestCase):
  def test_has_one_arm_for_the_gripper(self):
    self.assertEqual(OpentronsFlexBackend._num_arms, 1)

  def test_deck_frame_is_the_robot_frame(self):
    location = Coordinate(10, 20, 30)
    self.assertEqual(_flex_backend()._deck_to_robot_frame(location), location)

  def test_flex_pipette_volumes(self):
    backend = _flex_backend()
    self.assertEqual(backend.pipette_name2volume["p1000_single_flex"], 1000)
    self.assertEqual(backend.pipette_name2volume["p50_single_flex"], 50)
    self.assertEqual(backend.pipette_name2volume["flex_1channel_1000"], 1000)

  def test_pipette_table_has_no_bogus_200ul_entries(self):
    # the Flex ships no 200uL pipette; 96-channel is 1000uL only
    table = OpentronsFlexBackend.pipette_name2volume
    self.assertNotIn("p200_96_flex", table)
    self.assertNotIn("flex_96channel_200", table)
    self.assertNotIn(200, table.values())

  def test_tip_volume_compatibility(self):
    backend = _flex_backend()
    self.assertTrue(backend._tip_volume_supported(1000, 1000))
    self.assertTrue(backend._tip_volume_supported(1000, 50))
    self.assertFalse(backend._tip_volume_supported(50, 1000))

  def test_1000ul_tips_select_the_p1000_not_the_p50(self):
    backend = _flex_backend()
    tip = _tip(1000)
    self.assertFalse(backend.can_pick_up_tip(0, tip))  # p50 left
    self.assertTrue(backend.can_pick_up_tip(1, tip))  # p1000 right
    self.assertEqual(backend.select_tip_pipette(tip, with_tip=False), "R")

  def test_trash_resource_routes_to_trash(self):
    trash = FlexDeck().slots["A3"]
    assert trash is not None
    self.assertTrue(_flex_backend()._resource_is_trash(trash))


class FlexSerializeTests(unittest.TestCase):
  def test_serialize_roundtrips_host_and_port(self):
    with patch("ot_api.set_host"), patch("ot_api.set_port"):
      backend = OpentronsFlexBackend(host="1.2.3.4", port=31950)
    self.assertEqual(
      backend.serialize(),
      {"type": "OpentronsFlexBackend", "host": "1.2.3.4", "port": 31950},
    )


class FlexDeckConfigTests(unittest.IsolatedAsyncioTestCase):
  async def test_configure_deck_derives_fixtures_from_the_deck(self):
    backend = _flex_backend()
    backend.set_deck(FlexDeck())
    backend._request = MagicMock(return_value={})
    await backend._configure_deck()
    method, path, body = backend._request.call_args[0][:3]
    self.assertEqual((method, path), ("PUT", "/deck_configuration"))
    fixtures = body["data"]["cutoutFixtures"]
    self.assertIn({"cutoutId": "cutoutA3", "cutoutFixtureId": "trashBinAdapter"}, fixtures)
    self.assertIn({"cutoutId": "cutoutA1", "cutoutFixtureId": "singleLeftSlot"}, fixtures)
    self.assertIn({"cutoutId": "cutoutA2", "cutoutFixtureId": "singleCenterSlot"}, fixtures)
    self.assertIn({"cutoutId": "cutoutB3", "cutoutFixtureId": "singleRightSlot"}, fixtures)

  async def test_configure_deck_without_trash_frees_a3_as_a_slot(self):
    # the config must track the deck: with no trash, A3 is an ordinary right-column slot, not a bin
    backend = _flex_backend()
    backend.set_deck(FlexDeck(with_trash=False))
    backend._request = MagicMock(return_value={})
    await backend._configure_deck()
    fixtures = backend._request.call_args[0][2]["data"]["cutoutFixtures"]
    a3 = next(f for f in fixtures if f["cutoutId"] == "cutoutA3")
    self.assertEqual(a3["cutoutFixtureId"], "singleRightSlot")


class FlexPipettingFrameTests(unittest.IsolatedAsyncioTestCase):
  def _single_channel_backend(self) -> tuple[OpentronsFlexBackend, AsyncMock]:
    backend = _flex_backend()
    backend.set_deck(FlexDeck())
    backend.left_pipette = {"pipetteId": "L", "name": "p1000_single_flex"}
    backend.right_pipette = None
    backend.left_pipette_has_tip = True
    backend._ot = MagicMock()
    move_pipette_head = AsyncMock()
    backend.move_pipette_head = move_pipette_head
    return backend, move_pipette_head

  async def test_aspirate_moves_to_the_deck_frame_location_unchanged(self):
    # the Flex deck IS the robot frame, so unlike the OT-2 there is no slot-1 rebase
    backend, move_pipette_head = self._single_channel_backend()
    well = MagicMock()
    well.get_location_wrt.return_value = Coordinate(50.0, 60.0, 70.0)
    op = SingleChannelAspiration(
      resource=well, offset=Coordinate.zero(), tip=_tip(1000), volume=100.0,
      flow_rate=None, liquid_height=None, blow_out_air_volume=None, mix=None,
    )
    await backend.aspirate([op], [0])
    first_location = move_pipette_head.call_args_list[0].kwargs["location"]
    self.assertEqual(first_location, Coordinate(50.0, 60.0, 70.0))

  async def test_dispense_moves_to_the_deck_frame_location_unchanged(self):
    backend, move_pipette_head = self._single_channel_backend()
    well = MagicMock()
    well.get_location_wrt.return_value = Coordinate(12.0, 34.0, 56.0)
    op = SingleChannelDispense(
      resource=well, offset=Coordinate.zero(), tip=_tip(1000), volume=100.0,
      flow_rate=None, liquid_height=None, blow_out_air_volume=None, mix=None,
    )
    await backend.dispense([op], [0])
    first_location = move_pipette_head.call_args_list[0].kwargs["location"]
    self.assertEqual(first_location, Coordinate(12.0, 34.0, 56.0))


class FlexGripperTests(unittest.IsolatedAsyncioTestCase):
  async def test_pick_up_resource_defines_and_loads_movable_labware(self):
    backend = _flex_backend()
    deck = FlexDeck()
    backend.set_deck(deck)
    plate = Resource(name="myplate", size_x=127.0, size_y=85.0, size_z=14.0)
    deck.assign_child_at_slot(plate, "C2")
    backend._ot = MagicMock()
    backend._ot.labware.define.return_value = {"data": {"definitionUri": "pylabrobot/abc123/1"}}
    backend._run_command = MagicMock(return_value={})

    await backend.pick_up_resource(
      ResourcePickup(
        resource=plate, offset=Coordinate.zero(), pickup_distance_from_top=0,
        direction=GripDirection.FRONT,
      )
    )

    # a gripper-liftable labware definition was posted (has gripperOffsets)
    definition = backend._ot.labware.define.call_args[0][0]
    self.assertIn("gripperOffsets", definition)
    self.assertEqual(definition["dimensions"]["xDimension"], 127.0)

    # loadLabware used the slot NAME and an INT version (the Flex shape, not the OT-2 numeric slot)
    command_type, params = backend._run_command.call_args[0][0], backend._run_command.call_args[0][1]
    self.assertEqual(command_type, "loadLabware")
    self.assertEqual(params["location"], {"slotName": "C2"})
    self.assertIsInstance(params["version"], int)
    self.assertEqual(params["version"], 1)

    # the pickup is buffered until the drop (Opentrons moveLabware is atomic)
    assert backend._pending_pickup is not None
    self.assertEqual(backend._pending_pickup[0], backend._loaded_labware["myplate"])

  async def test_gripper_drop_emits_move_labware_using_gripper(self):
    backend = _flex_backend()
    deck = FlexDeck()
    backend.set_deck(deck)
    plate = Resource(name="plate", size_x=127.0, size_y=85.0, size_z=14.0)
    deck.assign_child_at_slot(plate, "C1")
    backend._loaded_labware = {"plate": "lw1"}
    backend._pending_pickup = ("lw1", plate)
    backend._run_command = MagicMock(return_value={})

    drop = ResourceDrop(
      resource=plate,
      destination=deck.slot_locations["B2"],
      destination_absolute_rotation=Rotation(0, 0, 0),
      offset=Coordinate.zero(),
      pickup_distance_from_top=0,
      pickup_direction=GripDirection.FRONT,
      direction=GripDirection.FRONT,
      rotation=0,
    )
    await backend.drop_resource(drop)

    backend._run_command.assert_called_once_with(
      "moveLabware",
      {"labwareId": "lw1", "newLocation": {"slotName": "B2"}, "strategy": "usingGripper"},
    )
    self.assertIsNone(backend._pending_pickup)

  async def test_gripper_drop_rejects_rotation(self):
    # the Flex gripper cannot rotate labware, so a rotating move must fail loudly, not silently
    backend = _flex_backend()
    deck = FlexDeck()
    backend.set_deck(deck)
    plate = Resource(name="plate", size_x=127.0, size_y=85.0, size_z=14.0)
    deck.assign_child_at_slot(plate, "C1")
    backend._pending_pickup = ("lw1", plate)
    backend._run_command = MagicMock(return_value={})
    drop = ResourceDrop(
      resource=plate,
      destination=deck.slot_locations["B2"],
      destination_absolute_rotation=Rotation(0, 0, 0),
      offset=Coordinate.zero(),
      pickup_distance_from_top=0,
      pickup_direction=GripDirection.FRONT,
      direction=GripDirection.LEFT,
      rotation=90,
    )
    with self.assertRaises(ValueError):
      await backend.drop_resource(drop)
    backend._run_command.assert_not_called()

  async def test_gripper_drop_without_pickup_raises(self):
    backend = _flex_backend()
    backend._pending_pickup = None
    drop = ResourceDrop(
      resource=Resource(name="p", size_x=1, size_y=1, size_z=1),
      destination=Coordinate.zero(),
      destination_absolute_rotation=Rotation(0, 0, 0),
      offset=Coordinate.zero(),
      pickup_distance_from_top=0,
      pickup_direction=GripDirection.FRONT,
      direction=GripDirection.FRONT,
      rotation=0,
    )
    with self.assertRaises(RuntimeError):
      await backend.drop_resource(drop)

  async def test_move_picked_up_resource_is_unsupported(self):
    backend = _flex_backend()
    move = ResourceMove(
      resource=Resource(name="p", size_x=1, size_y=1, size_z=1),
      location=Coordinate.zero(),
      gripped_direction=GripDirection.FRONT,
      pickup_distance_from_top=0,
      offset=Coordinate.zero(),
    )
    with self.assertRaises(NotImplementedError):
      await backend.move_picked_up_resource(move)


class FlexTrashTests(unittest.IsolatedAsyncioTestCase):
  def test_drop_tip_in_trash_targets_movable_trash_a3(self):
    backend = _flex_backend()
    backend._run_command = MagicMock(return_value={})
    backend._ot = MagicMock()
    backend._drop_tip_in_trash("R", 1.0, 2.0, 3.0)
    backend._run_command.assert_called_once_with(
      "moveToAddressableAreaForDropTip",
      {
        "pipetteId": "R",
        "addressableAreaName": "movableTrashA3",
        "offset": {"x": 1.0, "y": 2.0, "z": 3.0},
      },
    )
    backend._ot.lh.drop_tip_in_place.assert_called_once_with(pipette_id="R")


class FlexSetupTests(unittest.IsolatedAsyncioTestCase):
  async def test_setup_clears_stale_gripper_state(self):
    # a re-setup starts a new robot run, so labware ids buffered from a previous run must be dropped
    backend = _flex_backend()
    backend._loaded_labware = {"stale": "lw9"}
    backend._pending_pickup = ("lw9", Resource(name="stale", size_x=1, size_y=1, size_z=1))
    backend._ot = MagicMock()
    backend._ot.runs.create.return_value = "run1"
    backend._ot.lh.add_mounted_pipettes.return_value = (None, None)
    backend._ot.health.get.return_value = {"api_version": "8.0.0"}
    backend._configure_deck = AsyncMock()
    backend.home = AsyncMock()

    await backend.setup(skip_home=True)

    self.assertEqual(backend._loaded_labware, {})
    self.assertIsNone(backend._pending_pickup)
