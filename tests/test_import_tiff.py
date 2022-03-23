# Test that the import_tiff script works as intended

import unittest

class TestImportTiff(unittest.TestCase):

    def test_fail(self):
        self.assertFalse('start here')

if __name__ == '__main__':
    unittest.main()

