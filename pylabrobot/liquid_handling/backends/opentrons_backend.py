import inspect
import json
import logging
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from pylabrobot import utils
from pylabrobot.io import LOG_LEVEL_IO
from pylabrobot.liquid_handling.backends.backend import (
  LiquidHandlerBackend,
)
from pylabrobot.liquid_handling.errors import NoChannelError
from pylabrobot.liquid_handling.standard import (
  Drop,
  DropTipRack,
  MultiHeadAspirationContainer,
  MultiHeadAspirationPlate,
  MultiHeadDispenseContainer,
  MultiHeadDispensePlate,
  Pickup,
  PickupTipRack,
  ResourceDrop,
  ResourceMove,
  ResourcePickup,
  SingleChannelAspiration,
  SingleChannelDispense,
)
from pylabrobot.resources import (
  Coordinate,
  Resource,
  Tip,
)
from pylabrobot.resources.opentrons import OTDeck
from pylabrobot.resources.tip_rack import TipRack

try:
  import ot_api

  USE_OT = True
except ImportError as e:
  USE_OT = False
  _OT_IMPORT_ERROR = e


# https://github.com/Opentrons/opentrons/issues/14590
# https://labautomation.io/t/connect-pylabrobot-to-ot2/2862/18
_OT_DECK_IS_ADDRESSABLE_AREA_VERSION = "7.1.0"

logger = logging.getLogger(__name__)


def _version_tuple(version: str) -> Tuple[int, ...]:
  """Parse a dotted robot-software version into comparable integers.

  Comparing these as strings puts "10.0.0" below "7.1.0", so the version gate compares numerically.
  Each dotted segment contributes its leading integer ("0-beta" -> 0); a segment with no leading
  digit stops the parse. Only used to gate at coarse major.minor granularity, where the exact
  handling of a pre-release suffix does not change the outcome.
  """
  parts: List[int] = []
  for part in version.split("."):
    digits = ""
    for char in part:
      if not char.isdigit():
        break
      digits += char
    if digits == "":
      break
    parts.append(int(digits))
  return tuple(parts)


class _IOLogger:
  """Transparent proxy over the ``ot_api`` module that logs every call at
  ``LOG_LEVEL_IO``.

  Opentrons robots talk HTTP through ``ot_api`` rather than a pylabrobot.io transport, so
  this wrapper gives them the same wire-level logging every other backend gets from
  its io object. Submodules (``lh``, ``health``, ...) are wrapped recursively;
  plain attributes (e.g. ``run_id``) pass through untouched.
  """

  def __init__(self, target: Any, prefix: str = ""):
    self._target = target
    self._prefix = prefix

  def __getattr__(self, name: str) -> Any:
    attr = getattr(self._target, name)
    qualified = f"{self._prefix}.{name}" if self._prefix else name
    if inspect.ismodule(attr):
      return _IOLogger(attr, qualified)
    if callable(attr):

      def _logged(*args, **kwargs):
        parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        logger.log(LOG_LEVEL_IO, "%s(%s)", qualified, ", ".join(parts))
        return attr(*args, **kwargs)

      return _logged
    return attr


class OpentronsBackend(LiquidHandlerBackend):
  """Shared base for the Opentrons HTTP backends (OT-2 and Flex).

  Both robots run the same ``robot-server`` and are driven over the same HTTP API via ``ot_api``,
  so all of the transport, run lifecycle, JIT labware definition, pipette selection, and
  move-then-aspirate/dispense-in-place logic lives here. The parts that genuinely differ per robot
  (pipette catalog, deck-frame conversion, slot addressing, trash target, deck configuration, and
  the gripper) are override hooks implemented by the concrete subclasses.
  """

  # Subclasses fill this with the pipettes their robot reports and the tip volume each accepts.
  pipette_name2volume: Dict[str, int] = {}

  def __init__(self, host: str, port: int = 31950):
    super().__init__()

    if not USE_OT:
      raise RuntimeError(
        "Opentrons is not installed. Please run pip install pylabrobot[opentrons]."
        f" Import error: {_OT_IMPORT_ERROR}."
      )

    self.host = host
    self.port = port

    # A subclass (e.g. the chatterbox) can dry-run the backend by swapping this handle for a
    # recording stand-in; the real handle wraps ot_api to log every HTTP call at LOG_LEVEL_IO.
    self._ot: Any = _IOLogger(ot_api)

    self._ot.set_host(host)
    self._ot.set_port(port)

    self.ot_api_version: Optional[str] = None
    self.left_pipette: Optional[Dict[str, str]] = None
    self.right_pipette: Optional[Dict[str, str]] = None

    self.traversal_height = 120
    self._tip_racks: Dict[str, Union[int, str]] = {}  # tip_rack.name -> slot
    self._plr_name_to_load_name: Dict[str, str] = {}

  def serialize(self) -> dict:
    return {
      **super().serialize(),
      "host": self.host,
      "port": self.port,
    }

  async def setup(self, skip_home: bool = False):
    # create run
    run_id = self._ot.runs.create()
    self._ot.set_run(run_id)

    # tell the robot which fixtures are on the deck (Flex requires this; OT-2 is a no-op)
    await self._configure_deck()

    # get pipettes, then assign them
    self.left_pipette, self.right_pipette = self._ot.lh.add_mounted_pipettes()

    self.left_pipette_has_tip = self.right_pipette_has_tip = False

    # get api version
    health = self._ot.health.get()
    self.ot_api_version = health["api_version"]

    if not skip_home:
      await self.home()

  @property
  def num_channels(self) -> int:
    return len([p for p in [self.left_pipette, self.right_pipette] if p is not None])

  async def stop(self):
    """Cancel any active OT run, then clear labware definitions."""
    self._plr_name_to_load_name = {}
    self._tip_racks = {}
    self.left_pipette = None
    self.right_pipette = None

    # cancel the HTTP-API run if it exists (helpful to make device available again in official Opentrons app)
    run_id = getattr(self._ot, "run_id", None)
    if run_id:
      try:
        self._ot.requestor.post(f"/runs/{run_id}/cancel")
      except Exception:
        try:
          self._ot.requestor.post(f"/runs/{run_id}/actions/cancel")
        except Exception:
          try:
            self._ot.requestor.delete(f"/runs/{run_id}")
          except Exception:
            pass

  def _request(
    self, method: str, path: str, body: Optional[dict] = None, timeout: float = 60.0
  ) -> dict:
    """Make a raw HTTP request to the robot-server (for endpoints ot_api does not wrap)."""
    url = f"http://{self.host}:{self.port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Opentrons-Version": "3"}
    if data is not None:
      headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
      return cast(dict, json.loads(response.read().decode()))

  def _run_command(self, command_type: str, params: dict, timeout: float = 60.0) -> dict:
    """Enqueue a run command and block until it reaches a terminal status, raising on failure."""
    run_id = getattr(self._ot, "run_id", None)
    body = {"data": {"commandType": command_type, "params": params, "intent": "setup"}}
    path = f"/runs/{run_id}/commands?waitUntilComplete=true&timeout={int(timeout * 1000)}"
    result = cast(dict, self._request("POST", path, body, timeout=timeout + 5.0)["data"])
    if result.get("status") == "failed" or result.get("error"):
      raise RuntimeError(f"Opentrons {command_type} command failed: {result.get('error')}")
    return result

  def get_ot_name(self, plr_resource_name: str) -> str:
    """Opentrons only allows names in ^[a-z0-9._]+$, but in PLR we are flexible.
    So we map PLR names to OT names here.
    """
    if plr_resource_name not in self._plr_name_to_load_name:
      ot_load_name = uuid.uuid4().hex
      self._plr_name_to_load_name[plr_resource_name] = ot_load_name
    return self._plr_name_to_load_name[plr_resource_name]

  def select_tip_pipette(self, tip: Tip, with_tip: bool) -> Optional[str]:
    """Select a pipette based on maximum tip volume for tip pick up or drop.

    The volume of the head must match the maximum tip volume. If both pipettes have the same
    maximum volume, the left pipette is selected.

    Args:
      with_tip: If True, get a channel that has a tip.

    Returns:
      The id of the pipette, or None if no pipette is available.
    """

    if self.can_pick_up_tip(0, tip) and with_tip == self.left_pipette_has_tip:
      assert self.left_pipette is not None
      return cast(str, self.left_pipette["pipetteId"])

    if self.can_pick_up_tip(1, tip) and with_tip == self.right_pipette_has_tip:
      assert self.right_pipette is not None
      return cast(str, self.right_pipette["pipetteId"])

    return None

  def _build_tip_rack_definition(
    self, tip_rack: TipRack, tip: Tip, grip_distance_from_top: Optional[float] = None
  ) -> dict:
    ot_slot_size_y = 86
    definition: dict = {
      "schemaVersion": 2,
      "version": 1,
      "namespace": "pylabrobot",
      "metadata": {
        "displayName": self.get_ot_name(tip_rack.name),
        "displayCategory": "tipRack",
        "displayVolumeUnits": "µL",
      },
      "brand": {
        "brand": "unknown",
      },
      "parameters": {
        "format": "96Standard",
        "isTiprack": True,
        # should we get the tip length from calibration on the robot? /calibration/tip_length
        "tipLength": tip.total_tip_length,
        "tipOverlap": tip.fitting_depth,
        "loadName": self.get_ot_name(tip_rack.name),
        "isMagneticModuleCompatible": False,  # do we really care? If yes, store.
      },
      "ordering": utils.reshape_2d(
        [self.get_ot_name(tip_spot.name) for tip_spot in tip_rack.get_all_items()],
        (tip_rack.num_items_x, tip_rack.num_items_y),
      ),
      "cornerOffsetFromSlot": {
        "x": 0,
        "y": ot_slot_size_y
        - tip_rack.get_absolute_size_y(),  # hinges push it to the back (PLR is LFB, OT is LBB)
        "z": 0,
      },
      "dimensions": {
        "xDimension": tip_rack.get_absolute_size_x(),
        "yDimension": tip_rack.get_absolute_size_y(),
        "zDimension": tip_rack.get_absolute_size_z(),
      },
      "wells": {
        self.get_ot_name(child.name): {
          "depth": child.get_absolute_size_z(),
          "x": cast(Coordinate, child.location).x + child.get_absolute_size_x() / 2,
          "y": cast(Coordinate, child.location).y + child.get_absolute_size_y() / 2,
          "z": cast(Coordinate, child.location).z,
          "shape": "circular",
          "diameter": child.get_absolute_size_x(),
          "totalLiquidVolume": tip.maximal_volume,
        }
        for child in tip_rack.children
      },
      "groups": [
        {
          "wells": [self.get_ot_name(tip_spot.name) for tip_spot in tip_rack.get_all_items()],
          "metadata": {
            "displayName": None,
            "displayCategory": "tipRack",
            "wellBottomShape": "flat",  # required even for tip racks
          },
        }
      ],
    }
    if grip_distance_from_top is not None:
      definition["gripHeightFromLabwareBottom"] = max(
        0.0, tip_rack.get_absolute_size_z() - grip_distance_from_top
      )
    return definition

  async def _assign_tip_rack(
    self, tip_rack: TipRack, tip: Tip, grip_distance_from_top: Optional[float] = None
  ):
    lw = self._build_tip_rack_definition(tip_rack, tip, grip_distance_from_top)

    data = self._ot.labware.define(lw)
    namespace, definition, version = data["data"]["definitionUri"].split("/")

    labware_uuid = self.get_ot_name(tip_rack.name)
    slot = self._get_slot_for_resource(tip_rack)

    self._load_labware_at_slot(
      load_name=definition,
      namespace=namespace,
      version=version,
      slot=slot,
      labware_id=labware_uuid,
      display_name=self.get_ot_name(tip_rack.name),
    )

    self._tip_racks[tip_rack.name] = slot

  def _get_pickup_pipette(self, ops: List[Pickup]) -> str:
    """Get the pipette for a tip pick-up, or raise."""
    assert len(ops) == 1, "only one channel supported for now"
    op = ops[0]
    assert op.resource.parent is not None, "must not be a floating resource"
    pipette_id = self.select_tip_pipette(op.tip, with_tip=False)
    if not pipette_id:
      raise NoChannelError("No pipette channel of right type with no tip available.")
    return pipette_id

  def _get_drop_pipette(self, ops: List[Drop]) -> str:
    """Get the pipette for a tip drop, or raise."""
    assert len(ops) == 1, "only one channel supported for now"
    op = ops[0]
    assert op.resource.parent is not None, "must not be a floating resource"
    pipette_id = self.select_tip_pipette(op.tip, with_tip=True)
    if not pipette_id:
      raise NoChannelError("No pipette channel of right type with tip available.")
    return pipette_id

  def _get_liquid_pipette(
    self, ops: Union[List[SingleChannelAspiration], List[SingleChannelDispense]]
  ) -> str:
    """Get the pipette for an aspirate/dispense, or raise."""
    assert len(ops) == 1, "only one channel supported for now"
    pipette_id = self.select_liquid_pipette(ops[0].volume)
    if pipette_id is None:
      raise NoChannelError("No pipette channel of right type with tip available.")
    return pipette_id

  def _set_tip_state(self, pipette_id: str, has_tip: bool):
    """Update tip-mounted state for the pipette that was used.

    Validates ``pipette_id`` against both the left and right pipette configurations and updates
    state only if it matches a known, configured pipette; otherwise raises to avoid silently
    putting the backend into an inconsistent state.
    """
    if self.left_pipette is not None and pipette_id == self.left_pipette["pipetteId"]:
      self.left_pipette_has_tip = has_tip
      return

    if self.right_pipette is not None and pipette_id == self.right_pipette["pipetteId"]:
      self.right_pipette_has_tip = has_tip
      return

    raise ValueError(f"Unknown or unconfigured pipette_id {pipette_id!r} in _set_tip_state.")

  async def pick_up_tips(self, ops: List[Pickup], use_channels: List[int]):
    """Pick up tips from the specified resource."""

    pipette_id = self._get_pickup_pipette(ops)
    op = ops[0]

    offset_x, offset_y, offset_z = (
      op.offset.x,
      op.offset.y,
      op.offset.z,
    )

    # define tip rack JIT if it's not already assigned
    tip_rack = op.resource.parent
    assert isinstance(tip_rack, TipRack), "TipSpot's parent must be a TipRack."
    if tip_rack.name not in self._tip_racks:
      await self._assign_tip_rack(tip_rack, op.tip)

    offset_z += op.tip.total_tip_length

    self._ot.lh.pick_up_tip(
      labware_id=self.get_ot_name(tip_rack.name),
      well_name=self.get_ot_name(op.resource.name),
      pipette_id=pipette_id,
      offset_x=offset_x,
      offset_y=offset_y,
      offset_z=offset_z,
    )

    self._set_tip_state(pipette_id, True)

  async def drop_tips(self, ops: List[Drop], use_channels: List[int]):
    """Drop tips into a tip rack well or the robot's trash."""

    pipette_id = self._get_drop_pipette(ops)
    op = ops[0]

    offset_x, offset_y = op.offset.x, op.offset.y
    offset_z = op.offset.z + 10  # ad-hoc offset adjustment that makes it smoother

    if self._resource_is_trash(op.resource):
      self._drop_tip_in_trash(pipette_id, offset_x, offset_y, offset_z)
    else:
      tip_rack = op.resource.parent
      assert isinstance(tip_rack, TipRack), "TipSpot's parent must be a TipRack."
      if tip_rack.name not in self._tip_racks:
        await self._assign_tip_rack(tip_rack, op.tip)
      self._ot.lh.drop_tip(
        self.get_ot_name(tip_rack.name),
        well_name=self.get_ot_name(op.resource.name),
        pipette_id=pipette_id,
        offset_x=offset_x,
        offset_y=offset_y,
        offset_z=offset_z,
      )

    self._set_tip_state(pipette_id, False)

  def select_liquid_pipette(self, volume: float) -> Optional[str]:
    """Select a pipette based on volume for an aspiration or dispense.

    The volume of the tip mounted on the head must be greater than the volume to aspirate or
    dispense. If both pipettes have the same maximum volume, the left pipette is selected.

    Only heads with a tip are considered.

    Args:
      volume: The volume to aspirate or dispense.

    Returns:
      The id of the pipette, or None if no pipette is available.
    """

    if self.left_pipette is not None:
      left_volume = self.pipette_name2volume[self.left_pipette["name"]]
      if left_volume >= volume and self.left_pipette_has_tip:
        return cast(str, self.left_pipette["pipetteId"])

    if self.right_pipette is not None:
      right_volume = self.pipette_name2volume[self.right_pipette["name"]]
      if right_volume >= volume and self.right_pipette_has_tip:
        return cast(str, self.right_pipette["pipetteId"])

    return None

  def get_pipette_name(self, pipette_id: str) -> str:
    """Get the name of a pipette from its id."""

    if self.left_pipette is not None and pipette_id == self.left_pipette["pipetteId"]:
      return cast(str, self.left_pipette["name"])
    if self.right_pipette is not None and pipette_id == self.right_pipette["pipetteId"]:
      return cast(str, self.right_pipette["name"])
    raise ValueError(f"Unknown pipette id: {pipette_id}")

  async def aspirate(self, ops: List[SingleChannelAspiration], use_channels: List[int]):
    """Aspirate liquid from the specified resource using pip."""

    pipette_id = self._get_liquid_pipette(ops)
    op = ops[0]
    volume = op.volume

    pipette_name = self.get_pipette_name(pipette_id)
    flow_rate = op.flow_rate or self._get_default_aspiration_flow_rate(pipette_name)

    location = self._deck_to_robot_frame(
      op.resource.get_location_wrt(self.deck, "c", "c", "cavity_bottom")
      + op.offset
      + Coordinate(z=op.liquid_height or 0)
    )

    await self.move_pipette_head(
      location=location,
      minimum_z_height=self.traversal_height,
      pipette_id=pipette_id,
    )

    if op.mix is not None:
      for _ in range(op.mix.repetitions):
        self._ot.lh.aspirate_in_place(
          volume=op.mix.volume,
          flow_rate=op.mix.flow_rate,
          pipette_id=pipette_id,
        )
        self._ot.lh.dispense_in_place(
          volume=op.mix.volume,
          flow_rate=op.mix.flow_rate,
          pipette_id=pipette_id,
        )

    self._ot.lh.aspirate_in_place(
      volume=volume,
      flow_rate=flow_rate,
      pipette_id=pipette_id,
    )

    traversal_location = self._deck_to_robot_frame(
      op.resource.get_location_wrt(self.deck, "c", "c", "cavity_bottom") + op.offset
    )
    traversal_location.z = self.traversal_height
    await self.move_pipette_head(
      location=traversal_location,
      minimum_z_height=self.traversal_height,
      pipette_id=pipette_id,
    )

  async def dispense(self, ops: List[SingleChannelDispense], use_channels: List[int]):
    """Dispense liquid from the specified resource using pip."""

    pipette_id = self._get_liquid_pipette(ops)
    op = ops[0]
    volume = op.volume

    pipette_name = self.get_pipette_name(pipette_id)
    flow_rate = op.flow_rate or self._get_default_dispense_flow_rate(pipette_name)

    location = self._deck_to_robot_frame(
      op.resource.get_location_wrt(self.deck, "c", "c", "cavity_bottom")
      + op.offset
      + Coordinate(z=op.liquid_height or 0)
    )
    await self.move_pipette_head(
      location=location,
      minimum_z_height=self.traversal_height,
      pipette_id=pipette_id,
    )

    self._ot.lh.dispense_in_place(
      volume=volume,
      flow_rate=flow_rate,
      pipette_id=pipette_id,
    )

    if op.mix is not None:
      for _ in range(op.mix.repetitions):
        self._ot.lh.aspirate_in_place(
          volume=op.mix.volume,
          flow_rate=op.mix.flow_rate,
          pipette_id=pipette_id,
        )
        self._ot.lh.dispense_in_place(
          volume=op.mix.volume,
          flow_rate=op.mix.flow_rate,
          pipette_id=pipette_id,
        )

    traversal_location = self._deck_to_robot_frame(
      op.resource.get_location_wrt(self.deck, "c", "c", "cavity_bottom") + op.offset
    )
    traversal_location.z = self.traversal_height
    await self.move_pipette_head(
      location=traversal_location,
      minimum_z_height=self.traversal_height,
      pipette_id=pipette_id,
    )

  async def home(self):
    self._ot.health.home()

  async def pick_up_tips96(self, pickup: PickupTipRack):
    raise NotImplementedError("The Opentrons backend does not support the 96 head.")

  async def drop_tips96(self, drop: DropTipRack):
    raise NotImplementedError("The Opentrons backend does not support the 96 head.")

  async def aspirate96(
    self, aspiration: Union[MultiHeadAspirationPlate, MultiHeadAspirationContainer]
  ):
    raise NotImplementedError("The Opentrons backend does not support the 96 head.")

  async def dispense96(self, dispense: Union[MultiHeadDispensePlate, MultiHeadDispenseContainer]):
    raise NotImplementedError("The Opentrons backend does not support the 96 head.")

  async def pick_up_resource(self, pickup: ResourcePickup):
    raise NotImplementedError("This Opentrons backend does not support moving labware.")

  async def move_picked_up_resource(self, move: ResourceMove):
    raise NotImplementedError("This Opentrons backend does not support moving labware.")

  async def drop_resource(self, drop: ResourceDrop):
    raise NotImplementedError("This Opentrons backend does not support moving labware.")

  async def list_connected_modules(self) -> List[dict]:
    """List all connected temperature modules."""
    return cast(List[dict], self._ot.modules.list_connected_modules())

  def _pipette_id_for_channel(self, channel: int) -> str:
    pipettes = []
    if self.left_pipette is not None:
      pipettes.append(self.left_pipette["pipetteId"])
    if self.right_pipette is not None:
      pipettes.append(self.right_pipette["pipetteId"])
    if channel < 0 or channel >= len(pipettes):
      raise NoChannelError(f"Channel {channel} not available on this Opentrons setup.")
    return pipettes[channel]

  def _current_channel_position(self, channel: int) -> Tuple[str, Coordinate]:
    """Return the pipette id and its current position, in the deck frame.

    `savePosition` reports the pipette critical point in the robot frame, while callers work in
    the deck frame, so the pose is rebased on the way out.
    """

    pipette_id = self._pipette_id_for_channel(channel)
    result = self._run_command("savePosition", {"pipetteId": pipette_id})
    pos = result["result"]["position"]
    return pipette_id, self._robot_to_deck_frame(Coordinate(pos["x"], pos["y"], pos["z"]))

  async def get_channel_position(self, channel: int) -> Coordinate:
    """The channel's current deck-frame position (the mounted tip's end, or the nozzle if none).

    Public read for direct positioning: place a channel with move_channel_*, then read it back.
    Async to match the move_channel_* surface, though the underlying savePosition query is blocking.
    """
    _, position = self._current_channel_position(channel)
    return position

  async def prepare_for_manual_channel_operation(self, channel: int):
    """Validate channel exists (no-op otherwise)."""

    _ = self._pipette_id_for_channel(channel)

  async def _move_channel_axis(self, channel: int, axis: str, value: float):
    """Move one axis of a channel to an absolute deck-frame coordinate, holding the other two."""

    pipette_id, current = self._current_channel_position(channel)
    target = {"x": current.x, "y": current.y, "z": current.z}
    target[axis] = value
    await self.move_pipette_head(
      location=self._deck_to_robot_frame(Coordinate(**target)),
      minimum_z_height=self.traversal_height,
      pipette_id=pipette_id,
    )

  async def move_channel_x(self, channel: int, x: float):
    """Move a channel to an absolute x coordinate in the deck frame."""

    await self._move_channel_axis(channel, "x", x)

  async def move_channel_y(self, channel: int, y: float):
    """Move a channel to an absolute y coordinate in the deck frame."""

    await self._move_channel_axis(channel, "y", y)

  async def move_channel_z(self, channel: int, z: float):
    """Move a channel to an absolute z coordinate in the deck frame."""

    await self._move_channel_axis(channel, "z", z)

  async def move_pipette_head(
    self,
    location: Coordinate,
    speed: Optional[float] = None,
    minimum_z_height: Optional[float] = None,
    pipette_id: Optional[str] = None,
    force_direct: bool = False,
  ):
    """Move the pipette head to the specified location. When a tip is mounted, the location refers
    to the bottom of the tip. If no tip is mounted, the location refers to the bottom of the
    pipette head.

    Args:
      location: The location to move to.
      speed: The speed to move at, in mm/s.
      minimum_z_height: The minimum z height to move to. Appears to be broken in the Opentrons API.
      pipette_id: The id of the pipette to move. If `"left"` or `"right"`, the left or right
        pipette is used.
      force_direct: If True, move the pipette head directly in all dimensions.
    """

    if self.left_pipette is not None and pipette_id == "left":
      pipette_id = self.left_pipette["pipetteId"]
    elif self.right_pipette is not None and pipette_id == "right":
      pipette_id = self.right_pipette["pipetteId"]

    if pipette_id is None:
      raise ValueError("No pipette id given or left/right pipette not available.")

    self._ot.lh.move_arm(
      pipette_id=pipette_id,
      location_x=location.x,
      location_y=location.y,
      location_z=location.z,
      minimum_z_height=minimum_z_height,
      speed=speed,
      force_direct=force_direct,
    )

  def can_pick_up_tip(self, channel_idx: int, tip: Tip) -> bool:
    if channel_idx == 0:
      pipette = self.left_pipette
    elif channel_idx == 1:
      pipette = self.right_pipette
    else:
      return False
    if pipette is None:
      return False
    channel_volume = self.pipette_name2volume[pipette["name"]]
    return self._tip_volume_supported(channel_volume, tip.maximal_volume)

  # --- override hooks: implemented per concrete robot ---

  async def _configure_deck(self):
    """Declare the deck's fixtures to the robot before running. OT-2 needs nothing; Flex must."""

  def _deck_to_robot_frame(self, location: Coordinate) -> Coordinate:
    """Convert a deck-frame coordinate to the robot's motion frame."""
    raise NotImplementedError

  def _robot_to_deck_frame(self, location: Coordinate) -> Coordinate:
    """Convert a robot-frame coordinate to the deck frame. Inverse of `_deck_to_robot_frame`."""
    raise NotImplementedError

  def _get_default_aspiration_flow_rate(self, pipette_name: str) -> float:
    """Default aspiration flow rate in uL/s for the given pipette."""
    raise NotImplementedError

  def _get_default_dispense_flow_rate(self, pipette_name: str) -> float:
    """Default dispense flow rate in uL/s for the given pipette."""
    raise NotImplementedError

  def _tip_volume_supported(self, channel_volume: float, tip_volume: float) -> bool:
    """Whether a pipette of ``channel_volume`` can pick up a tip of ``tip_volume``."""
    raise NotImplementedError

  def _get_slot_for_resource(self, resource: Resource) -> Union[int, str]:
    """The robot's slot identifier for a resource that is placed on the deck."""
    raise NotImplementedError

  def _load_labware_at_slot(
    self,
    load_name: str,
    namespace: str,
    version: str,
    slot: Union[int, str],
    labware_id: str,
    display_name: str,
  ):
    """Load a defined labware onto ``slot`` on the robot."""
    raise NotImplementedError

  def _resource_is_trash(self, resource: Resource) -> bool:
    """Whether a tip-drop target resource should route to the robot's trash."""
    raise NotImplementedError

  def _drop_tip_in_trash(self, pipette_id: str, offset_x: float, offset_y: float, offset_z: float):
    """Drop the mounted tip into the robot's trash addressable area."""
    raise NotImplementedError


class OpentronsOT2Backend(OpentronsBackend):
  """Backend for the Opentrons OT-2 liquid handling robot."""

  pipette_name2volume = {
    "p10_single": 10,
    "p10_multi": 10,
    "p20_single_gen2": 20,
    "p20_multi_gen2": 20,
    "p50_single": 50,
    "p50_multi": 50,
    "p300_single": 300,
    "p300_multi": 300,
    "p300_single_gen2": 300,
    "p300_multi_gen2": 300,
    "p1000_single": 1000,
    "p1000_single_gen2": 1000,
    "p300_single_gen3": 300,
    "p1000_single_gen3": 1000,
  }

  def _deck_to_robot_frame(self, location: Coordinate) -> Coordinate:
    """Convert a deck-frame coordinate to the OT-2 robot frame.

    pylabrobot positions OT deck slots from the deck plate corner, whereas the OT-2 motion API
    expects coordinates in the robot frame whose origin is slot 1's corner. The two frames differ by
    slot 1's position in the deck frame, so subtract it.
    """
    return location - cast(OTDeck, self.deck).slot_locations[0]

  def _robot_to_deck_frame(self, location: Coordinate) -> Coordinate:
    return location + cast(OTDeck, self.deck).slot_locations[0]

  def _get_default_aspiration_flow_rate(self, pipette_name: str) -> float:
    """Get the default aspiration flow rate for the specified pipette in uL/s.

    Data from https://archive.ph/ZUN9f
    """

    return {
      "p300_multi_gen2": 94,
      "p10_single": 5,
      "p10_multi": 5,
      "p50_single": 25,
      "p50_multi": 25,
      "p300_single": 150,
      "p300_multi": 150,
      "p1000_single": 500,
      "p20_single_gen2": 3.78,
      "p300_single_gen2": 46.43,
      "p1000_single_gen2": 137.35,
      "p20_multi_gen2": 7.6,
    }[pipette_name]

  def _get_default_dispense_flow_rate(self, pipette_name: str) -> float:
    """Get the default dispense flow rate for the specified pipette in uL/s.

    Data from https://archive.ph/ZUN9f
    """

    return {
      "p300_multi_gen2": 94,
      "p10_single": 10,
      "p10_multi": 10,
      "p50_single": 50,
      "p50_multi": 50,
      "p300_single": 300,
      "p300_multi": 300,
      "p1000_single": 1000,
      "p20_single_gen2": 7.56,
      "p300_single_gen2": 92.86,
      "p1000_single_gen2": 274.7,
      "p20_multi_gen2": 7.6,
    }[pipette_name]

  def _tip_volume_supported(self, channel_volume: float, tip_volume: float) -> bool:
    if channel_volume == 20:
      return tip_volume in {10, 20}
    if channel_volume == 300:
      return tip_volume in {200, 300}
    if channel_volume == 1000:
      return tip_volume in {1000}
    raise ValueError(f"Unknown channel volume: {channel_volume}")

  def _get_slot_for_resource(self, resource: Resource) -> int:
    deck = resource.parent
    while deck is not None and not isinstance(deck, OTDeck):
      deck = deck.parent  # labware sits in a slot holder, whose parent is the deck
    assert isinstance(deck, OTDeck)
    slot = deck.get_slot(resource)
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
    self._ot.labware.add(
      load_name=load_name,
      namespace=namespace,
      ot_location=slot,
      version=version,
      labware_id=labware_id,
      display_name=display_name,
    )

  def _resource_is_trash(self, resource: Resource) -> bool:
    return (
      _version_tuple(cast(str, self.ot_api_version))
      >= _version_tuple(_OT_DECK_IS_ADDRESSABLE_AREA_VERSION)
      and resource.name == "trash"
    )

  def _drop_tip_in_trash(self, pipette_id: str, offset_x: float, offset_y: float, offset_z: float):
    self._ot.lh.move_to_addressable_area_for_drop_tip(
      pipette_id=pipette_id,
      offset_x=offset_x,
      offset_y=offset_y,
      offset_z=offset_z,
    )
    self._ot.lh.drop_tip_in_place(pipette_id=pipette_id)
