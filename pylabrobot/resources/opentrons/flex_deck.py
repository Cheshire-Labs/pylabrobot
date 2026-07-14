import textwrap
from typing import Dict, List, Optional, cast

from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.deck import Deck
from pylabrobot.resources.resource import Resource
from pylabrobot.resources.resource_holder import ResourceHolder
from pylabrobot.resources.trash import Trash

_SLOT_SIZE_X = 128.0
_SLOT_SIZE_Y = 86.0

# Front-left corner of every slot in the Flex robot frame (origin at slot D1), from shared-data
# ot3_standard.json. Columns 1-3 are pipette-reachable; the staging column 4 is gripper-only.
_FLEX_SLOT_LOCATIONS: Dict[str, Coordinate] = {
  "D1": Coordinate(0.0, 0.0, 0.0),
  "C1": Coordinate(0.0, 107.0, 0.0),
  "B1": Coordinate(0.0, 214.0, 0.0),
  "A1": Coordinate(0.0, 321.0, 0.0),
  "D2": Coordinate(164.0, 0.0, 0.0),
  "C2": Coordinate(164.0, 107.0, 0.0),
  "B2": Coordinate(164.0, 214.0, 0.0),
  "A2": Coordinate(164.0, 321.0, 0.0),
  "D3": Coordinate(328.0, 0.0, 0.0),
  "C3": Coordinate(328.0, 107.0, 0.0),
  "B3": Coordinate(328.0, 214.0, 0.0),
  "A3": Coordinate(328.0, 321.0, 0.0),
  "D4": Coordinate(492.0, 0.0, 14.5),
  "C4": Coordinate(492.0, 107.0, 14.5),
  "B4": Coordinate(492.0, 214.0, 14.5),
  "A4": Coordinate(492.0, 321.0, 14.5),
}

STAGING_SLOTS = frozenset({"A4", "B4", "C4", "D4"})

# The Flex default deck configuration mounts the movable trash bin at A3 (the trashBinAdapter
# fixture, which exposes the movableTrashA3 addressable area used for tip disposal).
_TRASH_SLOT = "A3"
_TRASH_SIZE_Z = 40.0


class FlexDeck(Deck):
  """The Opentrons Flex (OT-3) deck.

  Working slots A1-D3 and the gripper-only staging column A4-D4 are modeled as ResourceHolder
  children keyed by coordinate name; labware is placed with :meth:`assign_child_at_slot`. The deck
  frame is the Flex robot frame (origin at slot D1's front-left corner), so no deck-to-robot
  rebasing is needed. With ``with_trash`` the trash bin occupies A3, matching the Flex default deck
  configuration.
  """

  def __init__(
    self,
    size_x: float = 620.0,
    size_y: float = 407.0,
    size_z: float = 0,
    origin: Coordinate = Coordinate(0, 0, 0),
    with_trash: bool = True,
    name: str = "flex_deck",
    category: str = "deck",
  ):
    super().__init__(
      size_x=size_x, size_y=size_y, size_z=size_z, name=name, origin=origin, category=category
    )

    self._slot_holders: Dict[str, ResourceHolder] = {}
    for slot, base in _FLEX_SLOT_LOCATIONS.items():
      holder = ResourceHolder(
        name=f"{self.name}_slot_{slot}",
        size_x=_SLOT_SIZE_X,
        size_y=_SLOT_SIZE_Y,
        size_z=_TRASH_SIZE_Z if (slot == _TRASH_SLOT and with_trash) else 0,
      )
      self._slot_holders[slot] = holder
      super().assign_child_resource(holder, location=base)

    if with_trash:
      self._assign_trash()

  @property
  def slots(self) -> Dict[str, Optional[Resource]]:
    """The labware in each slot, or ``None`` for empty slots, keyed by slot name."""
    return {slot: holder.resource for slot, holder in self._slot_holders.items()}

  @property
  def slot_locations(self) -> Dict[str, Coordinate]:
    """The front-left corner of each slot in the robot frame, keyed by slot name."""
    return {slot: cast(Coordinate, holder.location) for slot, holder in self._slot_holders.items()}

  def _assign_trash(self):
    trash = Trash(name="trash", size_x=_SLOT_SIZE_X, size_y=_SLOT_SIZE_Y, size_z=_TRASH_SIZE_Z)
    self.assign_child_at_slot(trash, _TRASH_SLOT)

  def assign_child_resource(
    self,
    resource: Resource,
    location: Optional[Coordinate] = None,
    reassign: bool = True,
  ):
    """Assign a slot holder to the deck.

    The deck's direct children are the slot holders created in ``__init__``. Deserialization
    re-assigns those holders by name, replacing the placeholder with the loaded one. Labware is
    placed with :meth:`assign_child_at_slot`, not here.
    """

    existing = next((child for child in self.children if child.name == resource.name), None)
    if existing is not None:
      if not reassign:
        raise ValueError(f"Resource '{resource.name}' already assigned to deck")
      super().unassign_child_resource(existing)
      for slot, holder in self._slot_holders.items():
        if holder is existing:
          self._slot_holders[slot] = cast(ResourceHolder, resource)
          break
    elif not isinstance(resource, ResourceHolder):
      raise ValueError(
        f"Cannot assign '{resource.name}' directly to the deck. Use assign_child_at_slot to place "
        "labware into a slot."
      )

    super().assign_child_resource(resource, location=location, reassign=reassign)

  def assign_child_at_slot(self, resource: Resource, slot: str):
    if slot not in self._slot_holders:
      raise ValueError(f"Unknown Flex slot '{slot}'. Valid slots: {sorted(self._slot_holders)}")

    holder = self._slot_holders[slot]
    if holder.resource is not None:
      raise ValueError(f"Slot {slot} is already occupied")

    holder.assign_child_resource(resource)

  def unassign_child_resource(self, resource: Resource):
    for holder in self._slot_holders.values():
      if holder.resource is resource:
        holder.unassign_child_resource(resource)
        return
    super().unassign_child_resource(resource)

  def get_slot(self, resource: Resource) -> Optional[str]:
    """The slot name a resource is placed in, or ``None`` if it is not on the deck."""
    for slot, holder in self._slot_holders.items():
      if holder.resource is resource:
        return slot
    return None

  def get_slot_holder(self, slot: str) -> ResourceHolder:
    """The ResourceHolder for a slot, e.g. as a ``move_plate`` gripper destination."""
    if slot not in self._slot_holders:
      raise ValueError(f"Unknown Flex slot '{slot}'. Valid slots: {sorted(self._slot_holders)}")
    return self._slot_holders[slot]

  def get_slot_at_location(self, location: Coordinate, tolerance: float = 2.0) -> Optional[str]:
    """The slot whose front-left corner matches ``location`` (x/y within ``tolerance`` mm).

    The gripper receives a destination as a deck-frame coordinate (the LFB of the moved labware),
    never a slot name, so a move resolves its target slot by matching that corner here.
    """
    for slot, holder in self._slot_holders.items():
      corner = cast(Coordinate, holder.location)
      if abs(corner.x - location.x) <= tolerance and abs(corner.y - location.y) <= tolerance:
        return slot
    return None

  def summary(self) -> str:
    """An ASCII map of the deck, A-row (back) at top, staging column on the right."""

    def cell(slot: str) -> str:
      resource = self._slot_holders[slot].resource
      name = "Empty" if resource is None else resource.name
      if len(name) > 10:
        name = name[:8] + ".."
      return f"{slot}:{name}".ljust(13)

    rows = "".join(
      "\n      | " + " | ".join(cell(f"{row}{col}") for col in (1, 2, 3, 4)) + " |"
      for row in ("A", "B", "C", "D")
    )
    return textwrap.dedent(
      f"      Flex deck: {self.get_absolute_size_x()}mm x {self.get_absolute_size_y()}mm{rows}"
    )
