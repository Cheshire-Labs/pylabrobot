import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("ot_api")

from pylabrobot.liquid_handling.backends import OpentronsFlexBackend
from pylabrobot.liquid_handling.backends.opentrons_flex_backend import (
  _FLEX_ROBOT_COMMANDS_VERSION,
)
from pylabrobot.liquid_handling.standard import (
  GripDirection,
  MultiHeadAspirationContainer,
  MultiHeadAspirationPlate,
  ResourceDrop,
  ResourceMove,
  ResourcePickup,
  SingleChannelAspiration,
  SingleChannelDispense,
)
from pylabrobot.resources import Coordinate, Resource, Tip
from pylabrobot.resources.opentrons import FlexDeck
from pylabrobot.resources.plate import Plate
from pylabrobot.resources.well import Well
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


class FlexRobotCommandTests(unittest.IsolatedAsyncioTestCase):
  """The robot/* family: free-space axis motion and direct gripper jaw control.

  These are Flex-only. The robot-server runs an OT-3 hardware check on the axis commands and
  rejects them on an OT-2.
  """

  def setUp(self):
    self.backend = _flex_backend()
    self.backend.ot_api_version = _FLEX_ROBOT_COMMANDS_VERSION
    patcher = patch.object(
      self.backend, "_run_command", return_value={"result": {"position": {"extensionZ": 100.0}}}
    )
    self.run_command = patcher.start()
    self.addCleanup(patcher.stop)

  def _sent(self):
    command, params = self.run_command.call_args.args
    return command, params

  async def test_move_axes_to_sends_the_axis_map_verbatim(self):
    """The robot/ family takes snake_case params, unlike every other command on this API."""
    await self.backend.move_axes_to({"x": 10.0, "extensionZ": 100.0})
    command, params = self._sent()
    self.assertEqual(command, "robot/moveAxesTo")
    self.assertEqual(params, {"axis_map": {"x": 10.0, "extensionZ": 100.0}})

  async def test_move_axes_to_returns_the_reported_position(self):
    position = await self.backend.move_axes_to({"extensionZ": 100.0})
    self.assertEqual(position, {"extensionZ": 100.0})

  async def test_move_axes_to_omits_unset_optional_params(self):
    """An absent critical_point/speed must not be sent as null; the server defaults them."""
    await self.backend.move_axes_to({"x": 1.0})
    _, params = self._sent()
    self.assertNotIn("critical_point", params)
    self.assertNotIn("speed", params)

  async def test_move_axes_to_passes_critical_point_and_speed(self):
    await self.backend.move_axes_to({"x": 1.0}, critical_point={"x": 0.0}, speed=50.0)
    _, params = self._sent()
    self.assertEqual(params["critical_point"], {"x": 0.0})
    self.assertEqual(params["speed"], 50.0)

  async def test_move_axes_relative_sends_deltas(self):
    await self.backend.move_axes_relative({"extensionZ": -5.0})
    command, params = self._sent()
    self.assertEqual(command, "robot/moveAxesRelative")
    self.assertEqual(params, {"axis_map": {"extensionZ": -5.0}})

  async def test_move_axes_rejects_unknown_axis(self):
    """A typo'd axis would 422 at the server; fail early with the valid names instead."""
    with self.assertRaisesRegex(ValueError, "extensionZ"):
      await self.backend.move_axes_to({"gripperZ": 10.0})

  async def test_open_gripper_jaw(self):
    await self.backend.open_gripper_jaw()
    command, params = self._sent()
    self.assertEqual(command, "robot/openGripperJaw")
    self.assertEqual(params, {})

  async def test_close_gripper_jaw_sends_force(self):
    await self.backend.close_gripper_jaw(force=15.0)
    command, params = self._sent()
    self.assertEqual(command, "robot/closeGripperJaw")
    self.assertEqual(params, {"force": 15.0})

  async def test_close_gripper_jaw_omits_force_when_unset(self):
    """Without a force the robot applies its own default, so send no key at all."""
    await self.backend.close_gripper_jaw()
    _, params = self._sent()
    self.assertEqual(params, {})

  async def test_close_gripper_jaw_rejects_force_outside_the_gripper_range(self):
    for force in (1.0, 31.0):
      with self.subTest(force=force):
        with self.assertRaisesRegex(ValueError, "2.0"):
          await self.backend.close_gripper_jaw(force=force)

  async def test_robot_commands_require_8_3_0(self):
    self.backend.ot_api_version = "8.2.0"
    with self.assertRaisesRegex(RuntimeError, _FLEX_ROBOT_COMMANDS_VERSION):
      await self.backend.move_axes_to({"x": 1.0})
    self.run_command.assert_not_called()

  async def test_robot_commands_allowed_on_a_double_digit_major(self):
    """Version gating must compare numerically: "10.0.0" is newer than "8.3.0", but sorts
    before it as a string."""
    self.backend.ot_api_version = "10.0.0"
    await self.backend.move_axes_to({"x": 1.0})
    self.run_command.assert_called_once()

  async def test_robot_commands_allowed_on_a_dev_build(self):
    """A dev or simulator build reports "0.0.0.dev0" but runs current code, so it is capable
    even though its version number sorts below the gate."""
    self.backend.ot_api_version = "0.0.0.dev0"
    await self.backend.move_axes_to({"x": 1.0})
    self.run_command.assert_called_once()


class FlexBackendUnitTests(unittest.TestCase):
  def test_has_one_arm_for_the_gripper(self):
    self.assertEqual(OpentronsFlexBackend._num_arms, 1)

  def test_robot_frame_conversion_round_trips(self):
    location = Coordinate(10, 20, 30)
    backend = _flex_backend()
    self.assertEqual(backend._robot_to_deck_frame(backend._deck_to_robot_frame(location)), location)

  def test_deck_frame_is_the_robot_frame(self):
    location = Coordinate(10, 20, 30)
    self.assertEqual(_flex_backend()._deck_to_robot_frame(location), location)

  def test_flex_pipette_volumes(self):
    backend = _flex_backend()
    self.assertEqual(backend.pipette_name2volume["p1000_single_flex"], 1000)
    self.assertEqual(backend.pipette_name2volume["p50_single_flex"], 50)
    self.assertEqual(backend.pipette_name2volume["flex_1channel_1000"], 1000)

  def test_reported_96_channel_name_is_in_the_volume_table(self):
    # GET /pipettes reports the 96-channel as "p1000_96", not "p1000_96_flex"; the volume
    # lookup during setup KeyErrors if it is missing.
    self.assertEqual(OpentronsFlexBackend.pipette_name2volume["p1000_96"], 1000)

  def test_two_hand_pipettes_report_two_channels(self):
    backend = _flex_backend()  # left p50, right p1000
    self.assertFalse(backend._has_96_head)
    self.assertEqual(backend.num_channels, 2)

  def test_96_head_reports_96_channels(self):
    backend = _flex_backend()
    backend.left_pipette = {"pipetteId": "L", "name": "p1000_96"}
    backend.right_pipette = None
    self.assertTrue(backend._has_96_head)
    self.assertEqual(backend.num_channels, 96)

  def test_96_head_advertises_head96_installed(self):
    backend = _flex_backend()
    self.assertFalse(backend.head96_installed)  # two hand pipettes
    backend.left_pipette = {"pipetteId": "L", "name": "p1000_96"}
    backend.right_pipette = None
    self.assertTrue(backend.head96_installed)


class FlexWellReferencingTests(unittest.IsolatedAsyncioTestCase):
  """touch_tip / liquid_probe load the well's plate on demand, then reference the well."""

  def _backend_and_well(self):
    backend = _flex_backend()  # left p50, right p1000
    well = MagicMock(spec=Well)
    well.name = "plate_A2"
    plate = MagicMock(spec=Plate)
    plate.name = "plate"
    well.parent = plate
    return backend, well

  async def test_touch_tip_loads_the_plate_and_emits_touch_tip(self):
    backend, well = self._backend_and_well()
    with (
      patch.object(backend, "_assign_plate", new=AsyncMock()) as assign,
      patch.object(backend, "_run_command") as run,
    ):
      await backend.touch_tip(well, radius=0.5, use_channel=1)
    assign.assert_awaited_once_with(well.parent)
    command, params = run.call_args.args
    self.assertEqual(command, "touchTip")
    self.assertEqual(params["radius"], 0.5)
    self.assertEqual(params["pipetteId"], "R")

  async def test_try_liquid_probe_returns_none_when_no_liquid(self):
    backend, well = self._backend_and_well()
    with (
      patch.object(backend, "_assign_plate", new=AsyncMock()),
      patch.object(backend, "_run_command", return_value={"result": {"position": {"z": 3.0}}}),
    ):
      self.assertIsNone(await backend.try_liquid_probe(well, use_channel=1))

  async def test_liquid_probe_raises_when_no_liquid(self):
    backend, well = self._backend_and_well()
    with (
      patch.object(backend, "_assign_plate", new=AsyncMock()),
      patch.object(backend, "_run_command", return_value={"result": {"position": {"z": 3.0}}}),
    ):
      with self.assertRaisesRegex(RuntimeError, "no liquid"):
        await backend.liquid_probe(well, use_channel=1)

  async def test_liquid_probe_returns_z_when_liquid_found(self):
    backend, well = self._backend_and_well()
    with (
      patch.object(backend, "_assign_plate", new=AsyncMock()),
      patch.object(backend, "_run_command", return_value={"result": {"z_position": 5.5}}),
    ):
      self.assertEqual(await backend.liquid_probe(well, use_channel=1), 5.5)

  async def test_well_not_in_a_plate_is_rejected(self):
    backend, well = self._backend_and_well()
    well.parent = MagicMock()  # not a Plate
    with self.assertRaisesRegex(ValueError, "not part of a plate"):
      await backend.touch_tip(well, use_channel=1)


class Flex96PipettingTests(unittest.IsolatedAsyncioTestCase):
  def _backend_with_96(self):
    backend = _flex_backend()
    backend.left_pipette = {"pipetteId": "96id", "name": "p1000_96"}
    backend.right_pipette = None
    return backend

  async def test_96_ops_require_a_96_head(self):
    """The *96 methods refuse to run without a 96 head rather than mis-drive a hand pipette."""
    backend = _flex_backend()  # two hand pipettes, no 96
    with self.assertRaisesRegex(RuntimeError, "96-channel head"):
      backend._require_96_head()

  def test_ninety_six_target_is_a1_for_a_plate(self):
    backend = self._backend_with_96()
    a1, h12 = MagicMock(), MagicMock()
    plate_op = MagicMock(spec=MultiHeadAspirationPlate)
    plate_op.wells = [a1, h12]
    self.assertIs(backend._ninety_six_target(plate_op), a1)

  def test_ninety_six_target_is_the_container_for_a_reservoir(self):
    backend = self._backend_with_96()
    container = MagicMock()
    container_op = MagicMock(spec=MultiHeadAspirationContainer)
    container_op.container = container
    self.assertIs(backend._ninety_six_target(container_op), container)

  async def test_blow_out_in_place_uses_the_selected_channel(self):
    backend = self._backend_with_96()
    with patch.object(backend, "_run_command") as run:
      await backend.blow_out_in_place(flow_rate=50.0)
    command, params = run.call_args.args
    self.assertEqual(command, "blowOutInPlace")
    self.assertEqual(params, {"pipetteId": "96id", "flowRate": 50.0})

  async def test_unsafe_ungrip_labware(self):
    backend = self._backend_with_96()
    with patch.object(backend, "_run_command") as run:
      await backend.unsafe_ungrip_labware()
    command, params = run.call_args.args
    self.assertEqual(command, "unsafe/ungripLabware")
    self.assertEqual(params, {})

  async def test_unsafe_drop_tip_in_place(self):
    backend = self._backend_with_96()
    with patch.object(backend, "_run_command") as run:
      await backend.unsafe_drop_tip_in_place()
    command, params = run.call_args.args
    self.assertEqual(command, "unsafe/dropTipInPlace")
    self.assertEqual(params, {"pipetteId": "96id"})

  async def test_configure_nozzle_layout_all_sends_just_the_style(self):
    backend = self._backend_with_96()
    with patch.object(backend, "_run_command") as run:
      await backend.configure_nozzle_layout("ALL")
    _, params = run.call_args.args
    self.assertEqual(params["configurationParams"], {"style": "ALL"})

  async def test_configure_nozzle_layout_column_includes_primary_nozzle(self):
    backend = self._backend_with_96()
    with patch.object(backend, "_run_command") as run:
      await backend.configure_nozzle_layout("COLUMN", primary_nozzle="A1")
    _, params = run.call_args.args
    self.assertEqual(params["configurationParams"], {"style": "COLUMN", "primaryNozzle": "A1"})

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
      resource=well,
      offset=Coordinate.zero(),
      tip=_tip(1000),
      volume=100.0,
      flow_rate=None,
      liquid_height=None,
      blow_out_air_volume=None,
      mix=None,
    )
    await backend.aspirate([op], [0])
    first_location = move_pipette_head.call_args_list[0].kwargs["location"]
    self.assertEqual(first_location, Coordinate(50.0, 60.0, 70.0))

  async def test_dispense_moves_to_the_deck_frame_location_unchanged(self):
    backend, move_pipette_head = self._single_channel_backend()
    well = MagicMock()
    well.get_location_wrt.return_value = Coordinate(12.0, 34.0, 56.0)
    op = SingleChannelDispense(
      resource=well,
      offset=Coordinate.zero(),
      tip=_tip(1000),
      volume=100.0,
      flow_rate=None,
      liquid_height=None,
      blow_out_air_volume=None,
      mix=None,
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
        resource=plate,
        offset=Coordinate.zero(),
        pickup_distance_from_top=0,
        direction=GripDirection.FRONT,
      )
    )

    # a gripper-liftable labware definition was posted (has gripperOffsets)
    definition = backend._ot.labware.define.call_args[0][0]
    self.assertIn("gripperOffsets", definition)
    self.assertEqual(definition["dimensions"]["xDimension"], 127.0)

    # loadLabware used the slot NAME and an INT version (the Flex shape, not the OT-2 numeric slot)
    command_type, params = (
      backend._run_command.call_args[0][0],
      backend._run_command.call_args[0][1],
    )
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
