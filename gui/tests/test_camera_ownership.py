import unittest

from camera_ownership import CameraOwnershipCoordinator


class CameraOwnershipCoordinatorTests(unittest.TestCase):
    def test_only_one_page_can_own_the_camera(self):
        ownership = CameraOwnershipCoordinator()

        self.assertTrue(ownership.acquire("sorting"))
        self.assertEqual(ownership.owner, "sorting")
        self.assertFalse(ownership.acquire("acceptance"))
        self.assertFalse(ownership.release("acceptance"))
        self.assertTrue(ownership.release("sorting"))
        self.assertIsNone(ownership.owner)
        self.assertTrue(ownership.acquire("acceptance"))

    def test_same_page_can_acquire_camera_twice(self):
        ownership = CameraOwnershipCoordinator()

        self.assertTrue(ownership.acquire("acceptance"))
        self.assertTrue(ownership.acquire("acceptance"))


if __name__ == "__main__":
    unittest.main()
