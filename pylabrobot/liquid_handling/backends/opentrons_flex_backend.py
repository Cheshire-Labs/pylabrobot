from typing import Dict, List, Literal, Optional, Tuple, Union, cast

from pylabrobot.liquid_handling.backends.opentrons_backend import (
  OpentronsBackend,
  _version_tuple,
)
from pylabrobot.liquid_handling.standard import (
  DropTipRack,
  MultiHeadAspirationContainer,
  MultiHeadAspirationPlate,
  MultiHeadDispenseContainer,
  MultiHeadDispensePlate,
  PickupTipRack,
  ResourceDrop,
  ResourceMove,
  ResourcePickup,
)
from pylabrobot.resources import Coordinate, Resource
from pylabrobot.resources.opentrons import FlexDeck
from pylabrobot.resources.tip_rack import TipRack
from pylabrobot.resources.trash import Trash

# Addressable area exposed by the trashBinAdapter fixture at A3, used for tip disposal.
_TRASH_ADDRESSABLE_AREA = "movableTrashA3"

# The robot/* command family was added to the robot-server in software 8.3.0.
_FLEX_ROBOT_COMMANDS_VERSION = "8.3.0"

FlexMotorAxis = Literal[
  "x",
  "y",
  "leftZ",
  "rightZ",
  "leftPlunger",
  "rightPlunger",
  "extensionZ",
  "extensionJaw",
  "axis96ChannelCam",
]
"""Motor axes the robot/* commands address. `extensionZ` and `extensionJaw` are the gripper."""

_FLEX_MOTOR_AXES = frozenset(
  {
    "x",
    "y",
    "leftZ",
    "rightZ",
    "leftPlunger",
    "rightPlunger",
    "extensionZ",
    "extensionJaw",
    "axis96ChannelCam",
  }
)

# Grip force limits from shared-data/gripper/definitions/1/gripperV1.3.json.
_FLEX_GRIPPER_MIN_FORCE = 2.0
_FLEX_GRIPPER_MAX_FORCE = 30.0

_NINETY_SIX_CHANNEL_COUNT = 96


def _is_96_channel(pipette_name: str) -> bool:
  """Whether a reported pipette name denotes a 96-channel head (e.g. p1000_96, p200_96)."""
  return "96" in pipette_name


class OpentronsFlexBackend(OpentronsBackend):
  """Backend for the Opentrons Flex (OT-3) liquid handling robot.

  Extends the shared :class:`OpentronsBackend` with the Flex's pipette catalog, coordinate deck
  frame, string slot addressing (A1-D4), movable trash, deck configuration, and the Flex gripper
  (which the OT-2 lacks). Pair it with a :class:`~pylabrobot.resources.opentrons.FlexDeck`.
  """

  _num_arms = 1  # the Flex gripper

  pipette_name2volume = {
    # names the robot reports for attached pipettes (GET /pipettes). The 96-channel reports
    # "p1000_96" / "p200_96" (no _flex suffix), unlike the single/multi pipettes.
    "p50_single_flex": 50,
    "p50_multi_flex": 50,
    "p1000_single_flex": 1000,
    "p1000_multi_flex": 1000,
    "p1000_96": 1000,
    # loadPipette names
    "flex_1channel_50": 50,
    "flex_8channel_50": 50,
    "flex_1channel_1000": 1000,
    "flex_8channel_1000": 1000,
    "flex_96channel_1000": 1000,
  }

  def __init__(self, host: str, port: int = 31950):
    super().__init__(host, port)
    self._loaded_labware: Dict[str, str] = {}  # resource.name -> opentrons labware id
    self._pending_pickup: Optional[Tuple[str, Resource]] = None

  async def setup(self, skip_home: bool = False):
    await super().setup(skip_home=skip_home)
    self._loaded_labware = {}
    self._pending_pickup = None
    if self._has_96_head:
      # A 96-channel head must be told its nozzle layout before it will pipette; ALL selects the
      # full head. Callers re-configure for partial-column work via configure_nozzle_layout.
      await self.configure_nozzle_layout("ALL")

  @property
  def _has_96_head(self) -> bool:
    """Whether a 96-channel head is mounted. It is mutually exclusive with hand pipettes and,
    on the Flex, always reports on the left mount."""
    return self.left_pipette is not None and _is_96_channel(self.left_pipette["name"])

  @property
  def num_channels(self) -> int:
    if self._has_96_head:
      return _NINETY_SIX_CHANNEL_COUNT
    return super().num_channels

  @property
  def head96_installed(self) -> Optional[bool]:
    return self._has_96_head

  async def stop(self):
    await super().stop()
    self._loaded_labware = {}
    self._pending_pickup = None

  async def _configure_deck(self):
    self._request(
      "PUT", "/deck_configuration", {"data": {"cutoutFixtures": self._deck_configuration()}}
    )

  def _deck_configuration(self) -> List[Dict[str, str]]:
    """The Flex cutout fixtures implied by the paired FlexDeck's slots.

    Columns 1 and 2 are single slots; a column-3 cutout is the movable trash bin where the deck's
    trash sits and a single slot otherwise. Derived from the deck so the robot's fixtures cannot
    disagree with the deck model: FlexDeck(with_trash=False) frees A3 as an ordinary slot here too.
    """
    deck = self.deck
    assert isinstance(deck, FlexDeck), "OpentronsFlexBackend requires a FlexDeck."
    column_fixture = {1: "singleLeftSlot", 2: "singleCenterSlot", 3: "singleRightSlot"}
    config: List[Dict[str, str]] = []
    for slot, resource in deck.slots.items():
      column = int(slot[1])
      fixture = (
        "trashBinAdapter" if column == 3 and isinstance(resource, Trash) else column_fixture[column]
      )
      config.append({"cutoutId": f"cutout{slot}", "cutoutFixtureId": fixture})
    return config

  def _deck_to_robot_frame(self, location: Coordinate) -> Coordinate:
    # FlexDeck is defined directly in the robot frame (origin at slot D1), so no rebasing is needed.
    return location

  def _robot_to_deck_frame(self, location: Coordinate) -> Coordinate:
    return location

  def _get_default_aspiration_flow_rate(self, pipette_name: str) -> float:
    return {50: 35.0, 1000: 160.0}[self.pipette_name2volume[pipette_name]]

  def _get_default_dispense_flow_rate(self, pipette_name: str) -> float:
    return {50: 35.0, 1000: 160.0}[self.pipette_name2volume[pipette_name]]

  def _tip_volume_supported(self, channel_volume: float, tip_volume: float) -> bool:
    if channel_volume == 50:
      return tip_volume == 50
    if channel_volume == 1000:
      return tip_volume in {50, 200, 1000}
    raise ValueError(f"Unknown channel volume: {channel_volume}")

  def _find_flex_deck(self, resource: Resource) -> FlexDeck:
    deck = resource.parent
    while deck is not None and not isinstance(deck, FlexDeck):
      deck = deck.parent  # labware sits in a slot holder, whose parent is the deck
    assert isinstance(deck, FlexDeck), "resource must be on a FlexDeck"
    return deck

  def _get_slot_for_resource(self, resource: Resource) -> str:
    slot = self._find_flex_deck(resource).get_slot(resource)
    assert slot is not None, "resource must be on deck"
    return slot

  def _load_labware_at_slot(
    self,
    load_name: str,
    namespace: str,
    version: str,
    slot: Union[int, str],
    labware_id: str,
    display_name: str,
  ):
    # ot_api.labware.add reads a str location as a moduleId, so the Flex loads by slot name here.
    self._run_command(
      "loadLabware",
      {
        "location": {"slotName": slot},
        "loadName": load_name,
        "namespace": namespace,
        "version": int(version),
        "labwareId": labware_id,
        "displayName": display_name,
      },
    )

  def _resource_is_trash(self, resource: Resource) -> bool:
    return resource.name == "trash"

  def _drop_tip_in_trash(self, pipette_id: str, offset_x: float, offset_y: float, offset_z: float):
    self._run_command(
      "moveToAddressableAreaForDropTip",
      {
        "pipetteId": pipette_id,
        "addressableAreaName": _TRASH_ADDRESSABLE_AREA,
        "offset": {"x": offset_x, "y": offset_y, "z": offset_z},
      },
    )
    self._ot.lh.drop_tip_in_place(pipette_id=pipette_id)

  # --- gripper: PLR's pick/move/drop maps onto Opentrons' atomic moveLabware ---

  def _build_movable_labware_definition(self, resource: Resource) -> dict:
    name = self.get_ot_name(resource.name)
    size_x = resource.get_absolute_size_x()
    size_y = resource.get_absolute_size_y()
    size_z = resource.get_absolute_size_z()
    return {
      "schemaVersion": 2,
      "version": 1,
      "namespace": "pylabrobot",
      "metadata": {
        "displayName": name,
        "displayCategory": "wellPlate",
        "displayVolumeUnits": "µL",
      },
      "brand": {"brand": "unknown"},
      "parameters": {
        "format": "irregular",
        "isTiprack": False,
        "loadName": name,
        "isMagneticModuleCompatible": False,
      },
      "ordering": [["A1"]],
      "cornerOffsetFromSlot": {"x": 0, "y": 0, "z": 0},
      "dimensions": {"xDimension": size_x, "yDimension": size_y, "zDimension": size_z},
      "wells": {
        "A1": {
          "depth": 0,
          "x": size_x / 2,
          "y": size_y / 2,
          "z": 0,
          "shape": "circular",
          "diameter": 5,
          "totalLiquidVolume": 0,
        }
      },
      "groups": [{"wells": ["A1"], "metadata": {"wellBottomShape": "flat"}}],
      # required for the gripper to pick the labware up
      "gripperOffsets": {
        "default": {
          "pickUpOffset": {"x": 0, "y": 0, "z": 0},
          "dropOffset": {"x": 0, "y": 0, "z": 0},
        }
      },
    }

  def _define_and_load_movable_labware(self, resource: Resource) -> str:
    definition = self._build_movable_labware_definition(resource)
    data = self._ot.labware.define(definition)
    namespace, load_name, version = data["data"]["definitionUri"].split("/")
    labware_id = self.get_ot_name(resource.name)
    self._load_labware_at_slot(
      load_name=load_name,
      namespace=namespace,
      version=version,
      slot=self._get_slot_for_resource(resource),
      labware_id=labware_id,
      display_name=self.get_ot_name(resource.name),
    )
    return labware_id

  # --- robot/*: free-space axis motion and direct gripper control (Flex only) ---

  def _require_robot_commands(self, command: str) -> None:
    version = self.ot_api_version
    if version is None:
      raise RuntimeError(f"{command} requires setup() to have run, to read the robot's version.")
    # A dev or simulator build reports "0.0.0.dev0" but runs current code, so it supports every
    # command; only gate released builds by their version number.
    if "dev" in version:
      return
    if _version_tuple(version) < _version_tuple(_FLEX_ROBOT_COMMANDS_VERSION):
      raise RuntimeError(
        f"{command} requires Opentrons robot software {_FLEX_ROBOT_COMMANDS_VERSION} or newer, "
        f"but this robot reports {version}."
      )

  def _check_axes(self, axis_map: Dict[str, float]) -> None:
    unknown = sorted(set(axis_map) - _FLEX_MOTOR_AXES)
    if unknown:
      raise ValueError(f"Unknown motor axes {unknown}. Valid axes: {sorted(_FLEX_MOTOR_AXES)}.")

  def _move_axes(
    self,
    command: str,
    axis_map: Dict[str, float],
    critical_point: Optional[Dict[str, float]] = None,
    speed: Optional[float] = None,
  ) -> Dict[str, float]:
    self._require_robot_commands(command)
    self._check_axes(axis_map)
    # The robot/* commands take snake_case params, unlike the rest of this API.
    params: dict = {"axis_map": axis_map}
    if critical_point is not None:
      params["critical_point"] = critical_point
    if speed is not None:
      params["speed"] = speed
    result = self._run_command(command, params)
    return cast(Dict[str, float], result["result"]["position"])

  async def move_axes_to(
    self,
    axis_map: Dict[str, float],
    critical_point: Optional[Dict[str, float]] = None,
    speed: Optional[float] = None,
  ) -> Dict[str, float]:
    """Move the named axes to absolute positions, in mm. Axes not named are held.

    Args:
      axis_map: Target position per axis, keyed by `FlexMotorAxis`.
      critical_point: The point on the mounted tool the target refers to.
      speed: Travel speed in mm/s.

    Returns:
      The position of every axis after the move.
    """

    return self._move_axes("robot/moveAxesTo", axis_map, critical_point, speed)

  async def move_axes_relative(
    self, axis_map: Dict[str, float], speed: Optional[float] = None
  ) -> Dict[str, float]:
    """Move the named axes by a signed delta, in mm. Axes not named are held.

    Returns:
      The position of every axis after the move.
    """

    return self._move_axes("robot/moveAxesRelative", axis_map, speed=speed)

  async def open_gripper_jaw(self) -> None:
    """Open the gripper jaw, homing it."""

    self._require_robot_commands("robot/openGripperJaw")
    self._run_command("robot/openGripperJaw", {})

  async def close_gripper_jaw(self, force: Optional[float] = None) -> None:
    """Close the gripper jaw.

    Args:
      force: Grip force in Newtons. The robot applies its own default when this is None. There is
        no jaw-width parameter; drive the `extensionJaw` axis to command a width.
    """

    self._require_robot_commands("robot/closeGripperJaw")
    params: dict = {}
    if force is not None:
      if not _FLEX_GRIPPER_MIN_FORCE <= force <= _FLEX_GRIPPER_MAX_FORCE:
        raise ValueError(
          f"Grip force must be between {_FLEX_GRIPPER_MIN_FORCE} and {_FLEX_GRIPPER_MAX_FORCE} "
          f"Newtons, got {force}."
        )
      params["force"] = force
    self._run_command("robot/closeGripperJaw", params)

  # --- 96-channel head pipetting (valid only when a 96 head is mounted) ---

  async def configure_nozzle_layout(
    self,
    style: Literal["ALL", "SINGLE", "ROW", "COLUMN", "QUADRANT"] = "ALL",
    primary_nozzle: Optional[str] = None,
    front_right_nozzle: Optional[str] = None,
    back_left_nozzle: Optional[str] = None,
  ) -> None:
    """Select which nozzles of the 96-head are active.

    ``ALL`` uses the whole head. ``SINGLE``/``ROW``/``COLUMN``/``QUADRANT`` select a subset for
    partial pickup and pipetting; ``primary_nozzle`` is the anchor corner (one of A1, H1, A12,
    H12), and ``QUADRANT`` additionally needs ``front_right_nozzle`` and ``back_left_nozzle``.
    """
    pipette_id = self._require_96_head()
    config: dict = {"style": style}
    if primary_nozzle is not None:
      config["primaryNozzle"] = primary_nozzle
    if front_right_nozzle is not None:
      config["frontRightNozzle"] = front_right_nozzle
    if back_left_nozzle is not None:
      config["backLeftNozzle"] = back_left_nozzle
    self._run_command(
      "configureNozzleLayout", {"pipetteId": pipette_id, "configurationParams": config}
    )

  def _require_96_head(self) -> str:
    if not self._has_96_head:
      raise RuntimeError("The *96 operations require a 96-channel head, which is not mounted.")
    assert self.left_pipette is not None
    return self.left_pipette["pipetteId"]

  async def pick_up_tips96(self, pickup: PickupTipRack):
    """Pick up a full rack of tips with the 96-channel head.

    With the ALL nozzle layout the head references the rack's A1 well and engages all 96 tips.
    """
    pipette_id = self._require_96_head()
    tip_rack = pickup.resource
    tip = next((t for t in pickup.tips if t is not None), None)
    if tip is None:
      raise ValueError("pick_up_tips96 needs at least one tip in the rack.")
    if tip_rack.name not in self._tip_racks:
      await self._assign_tip_rack(tip_rack, tip)
    a1 = tip_rack.get_item("A1")
    self._ot.lh.pick_up_tip(
      labware_id=self.get_ot_name(tip_rack.name),
      well_name=self.get_ot_name(a1.name),
      pipette_id=pipette_id,
      offset_x=pickup.offset.x,
      offset_y=pickup.offset.y,
      offset_z=pickup.offset.z + tip.total_tip_length,
    )
    self._set_tip_state(pipette_id, True)

  async def drop_tips96(self, drop: DropTipRack):
    """Drop the 96-head tips into the trash, or back into a rack that is already loaded."""
    pipette_id = self._require_96_head()
    offset_z = drop.offset.z + 10  # matches the single-channel drop's smoothing offset
    resource = drop.resource
    if isinstance(resource, TipRack) and not self._resource_is_trash(resource):
      if resource.name not in self._tip_racks:
        raise RuntimeError(
          f"Cannot drop 96 tips into rack {resource.name!r}: it is not loaded on the robot."
        )
      a1 = resource.get_item("A1")
      self._ot.lh.drop_tip(
        self.get_ot_name(resource.name),
        well_name=self.get_ot_name(a1.name),
        pipette_id=pipette_id,
        offset_x=drop.offset.x,
        offset_y=drop.offset.y,
        offset_z=offset_z,
      )
    else:
      self._drop_tip_in_trash(pipette_id, drop.offset.x, drop.offset.y, offset_z)
    self._set_tip_state(pipette_id, False)

  def _ninety_six_target(
    self,
    op: Union[
      MultiHeadAspirationPlate,
      MultiHeadAspirationContainer,
      MultiHeadDispensePlate,
      MultiHeadDispenseContainer,
    ],
  ) -> Resource:
    """The resource the head references. For a plate the head aligns nozzle A1 to well A1."""
    if isinstance(op, (MultiHeadAspirationPlate, MultiHeadDispensePlate)):
      return op.wells[0]
    return op.container

  async def _move_96_head_over(
    self, target: Resource, offset: Coordinate, liquid_height: float, pipette_id: str
  ) -> None:
    location = self._deck_to_robot_frame(
      target.get_location_wrt(self.deck, "c", "c", "cavity_bottom")
      + offset
      + Coordinate(z=liquid_height)
    )
    await self.move_pipette_head(
      location=location, minimum_z_height=self.traversal_height, pipette_id=pipette_id
    )

  async def _retract_96_head(self, target: Resource, offset: Coordinate, pipette_id: str) -> None:
    up = self._deck_to_robot_frame(
      target.get_location_wrt(self.deck, "c", "c", "cavity_bottom") + offset
    )
    up.z = self.traversal_height
    await self.move_pipette_head(
      location=up, minimum_z_height=self.traversal_height, pipette_id=pipette_id
    )

  async def aspirate96(
    self, aspiration: Union[MultiHeadAspirationPlate, MultiHeadAspirationContainer]
  ):
    """Aspirate from a whole plate (or reservoir) with the 96-channel head."""
    pipette_id = self._require_96_head()
    target = self._ninety_six_target(aspiration)
    flow_rate = aspiration.flow_rate or self._get_default_aspiration_flow_rate(
      self.get_pipette_name(pipette_id)
    )
    await self._move_96_head_over(
      target, aspiration.offset, aspiration.liquid_height or 0, pipette_id
    )
    self._ot.lh.aspirate_in_place(
      volume=aspiration.volume, flow_rate=flow_rate, pipette_id=pipette_id
    )
    await self._retract_96_head(target, aspiration.offset, pipette_id)

  async def dispense96(self, dispense: Union[MultiHeadDispensePlate, MultiHeadDispenseContainer]):
    """Dispense to a whole plate (or reservoir) with the 96-channel head."""
    pipette_id = self._require_96_head()
    target = self._ninety_six_target(dispense)
    flow_rate = dispense.flow_rate or self._get_default_dispense_flow_rate(
      self.get_pipette_name(pipette_id)
    )
    await self._move_96_head_over(target, dispense.offset, dispense.liquid_height or 0, pipette_id)
    self._ot.lh.dispense_in_place(
      volume=dispense.volume, flow_rate=flow_rate, pipette_id=pipette_id
    )
    await self._retract_96_head(target, dispense.offset, pipette_id)

  async def pick_up_resource(self, pickup: ResourcePickup):
    resource = pickup.resource
    if resource.name not in self._loaded_labware:
      self._loaded_labware[resource.name] = self._define_and_load_movable_labware(resource)
    self._pending_pickup = (self._loaded_labware[resource.name], resource)

  async def move_picked_up_resource(self, move: ResourceMove):
    raise NotImplementedError(
      "The Flex gripper moves labware atomically; intermediate waypoints are not supported."
    )

  async def drop_resource(self, drop: ResourceDrop):
    if self._pending_pickup is None:
      raise RuntimeError("drop_resource called without a preceding pick_up_resource.")
    if drop.rotation != 0:
      raise ValueError(
        "The Flex gripper cannot rotate labware; pickup_direction and drop_direction must match "
        f"(the requested move rotates the labware {drop.rotation} degrees)."
      )
    labware_id, _ = self._pending_pickup
    slot = cast(FlexDeck, self.deck).get_slot_at_location(drop.destination)
    if slot is None:
      raise ValueError(f"No Flex deck slot matches gripper destination {drop.destination}.")
    self._run_command(
      "moveLabware",
      {
        "labwareId": labware_id,
        "newLocation": {"slotName": slot},
        "strategy": "usingGripper",
      },
    )
    self._pending_pickup = None
