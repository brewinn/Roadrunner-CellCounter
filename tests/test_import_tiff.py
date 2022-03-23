# Test that the import_tiff script works as intended

import unittest

# For finding the image in the resources directory
#  regardless of the current working directory
import os 

# Import the relevant function after installing the package
from cell_counter.import_tiff import tiff_to_array

class TestImportTiff(unittest.TestCase):

    def test_tiff_page_count(self):
        test_dir = os.path.dirname(__file__)
        path = test_dir + '/../resources/multipage_tiff_example.tif'
        images = tiff_to_array(path)
        self.assertEqual(len(images), 10)


if __name__ == '__main__':
    unittest.main()

