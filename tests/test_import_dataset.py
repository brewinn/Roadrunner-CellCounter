# Test that the import_dataset script works as intended

import unittest

# For finding the image in the resources directory
#  regardless of the current working directory
#import os

# Import the relevant function after installing the package
#from cell_counter.import_dataset import load_dataset


class TestImportDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_fail(self):
        self.assertEqual('fail!', 'start here')


if __name__ == "__main__":
    unittest.main()

