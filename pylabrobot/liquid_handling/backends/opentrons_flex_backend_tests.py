import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ot_api")

from pylabrobot.liquid_handling.backends import OpentronsFlexBackend
from pylabrobot.liquid_handling.standard import GripDirection, ResourceDrop
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
  return backend


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

  def test_tip_volume_compatibility(self):
    backend = _flex_backend()
    self.assertTrue(backend._tip_volume_supported(1000, 1000))
    self.assertTrue(backend._tip_volume_supported(1000, 50))
    self.assertFalse(backend._tip_volume_supported(50, 1000))

  def test_1000ul_tips_select_the_p1000_not_the_p50(self):
    backend = _flex_backend()
    tip = Tip(
      name="tip", has_filter=False, total_tip_length=95.0, maximal_volume=1000, fitting_depth=8.0
    )
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


class FlexCommandShapeTests(unittest.IsolatedAsyncioTestCase):
  async def test_configure_deck_puts_flex_config_with_trash_at_a3(self):
    backend = _flex_backend()
    backend._request = MagicMock(return_value={})
    await backend._configure_deck()
    method, path, body = backend._request.call_args[0][:3]
    self.assertEqual((method, path), ("PUT", "/deck_configuration"))
    fixtures = body["data"]["cutoutFixtures"]
    self.assertIn({"cutoutId": "cutoutA3", "cutoutFixtureId": "trashBinAdapter"}, fixtures)

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
