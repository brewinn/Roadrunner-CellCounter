# Test that the import_tiff script works as intended

import unittest

# For finding the image in the resources directory
#  regardless of the current working directory
import os 

# Import the relevant function after installing the package
from cell_counter.import_tiff import tiff_to_array

class TestImportTiff(unittest.TestCase):

    def setUp(self):
        test_dir = os.path.dirname(__file__)
        path = test_dir + '/../resources/multipage_tiff_example.tif'
        self.images = tiff_to_array(path)


    def test_tiff_page_count(self):
        self.assertEqual(len(self.images), 10)

    def test_tiff_cell_values(self):
        for index, image in enumerate(self.images):
            #print(f'{index=}:\n{image=}')
            self.assertEqual(image[0, 0], 255)
            # Check for black pixel on the zero of page 10
            #  note that the coordinates start at the top left,
            #  and the coordinates are [y,x] rather than [x,y].
            self.assertEqual(image[440, 550], 2 if index == 9 else 255)



if __name__ == '__main__':
    unittest.main()

