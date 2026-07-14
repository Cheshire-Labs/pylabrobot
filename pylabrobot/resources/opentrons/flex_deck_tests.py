import unittest

from pylabrobot.resources import Coordinate, Resource
from pylabrobot.resources.opentrons import FlexDeck


class FlexDeckTests(unittest.TestCase):
  def test_has_16_slots_with_trash_at_a3(self):
    deck = FlexDeck()
    self.assertEqual(len(deck.slots), 16)
    self.assertIsNotNone(deck.slots["A3"])
    self.assertEqual(deck.slots["A3"].name, "trash")
    self.assertIsNone(deck.slots["A1"])

  def test_d1_is_the_robot_origin(self):
    deck = FlexDeck()
    self.assertEqual(deck.slot_locations["D1"], Coordinate(0.0, 0.0, 0.0))
    self.assertEqual(deck.slot_locations["A1"], Coordinate(0.0, 321.0, 0.0))
    self.assertEqual(deck.slot_locations["D3"], Coordinate(328.0, 0.0, 0.0))

  def test_staging_column_present_and_raised(self):
    deck = FlexDeck()
    for slot in ("A4", "B4", "C4", "D4"):
      self.assertIn(slot, deck.slots)
      self.assertEqual(deck.slot_locations[slot].z, 14.5)

  def test_assign_and_get_slot(self):
    deck = FlexDeck()
    plate = Resource(name="plate", size_x=127.0, size_y=85.0, size_z=14.0)
    deck.assign_child_at_slot(plate, "C2")
    self.assertIs(deck.slots["C2"], plate)
    self.assertEqual(deck.get_slot(plate), "C2")

  def test_occupied_slot_raises(self):
    deck = FlexDeck()
    deck.assign_child_at_slot(Resource(name="a", size_x=1, size_y=1, size_z=1), "B1")
    with self.assertRaises(ValueError):
      deck.assign_child_at_slot(Resource(name="b", size_x=1, size_y=1, size_z=1), "B1")

  def test_unknown_slot_raises(self):
    deck = FlexDeck()
    with self.assertRaises(ValueError):
      deck.assign_child_at_slot(Resource(name="x", size_x=1, size_y=1, size_z=1), "E9")

  def test_get_slot_at_location_reverse_lookup(self):
    deck = FlexDeck()
    # the gripper resolves a destination coordinate (the LFB corner) back to a slot name
    self.assertEqual(deck.get_slot_at_location(Coordinate(328.0, 214.0, 0.0)), "B3")
    self.assertEqual(deck.get_slot_at_location(Coordinate(328.5, 214.3, 0.0)), "B3")  # tolerance
    self.assertIsNone(deck.get_slot_at_location(Coordinate(50.0, 50.0, 0.0)))
