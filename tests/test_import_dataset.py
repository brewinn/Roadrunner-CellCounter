# Test that the import_dataset script works as intended

import unittest

# Import the relevant function after installing the package
from cell_counter.import_dataset import load_synthetic_dataset


class TestImportSyntheticDataset(unittest.TestCase):
    def setUp(self):
        # Import dataset 
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = load_synthetic_dataset(seed = 1, num = 100)

    def test_dataset_size(self):
        self.assertEqual(len(self.train_images), 80)
        self.assertEqual(len(self.test_images), 20)

    def test_bad_size(self):
        self.assertRaises(Exception, load_synthetic_dataset, **{'num': -1})
        self.assertRaises(Exception, load_synthetic_dataset, **{'num': 100000000000})

    def test_imported_images(self):
        # With the seed set to 1, the first three images in the training set
        # should be SIMCEPImages_P18_C74_F48_s03_w1.TIF,
        # SIMCEPImages_D01_C1_F10_s17_w1.TIF, and
        # SIMCEPImages_J15_C61_F29_s05_w2.TIF

        # Test for SIMCEPImages_P18_C74_F48_s03_w1.TIF
        self.assertEqual(self.train_images[0][170, 350], 1)
        self.assertEqual(self.train_images[0][170, 400], 111)


        # Test for SIMCEPImages_D01_C1_F10_s17_w1.TIF
        self.assertEqual(self.train_images[1][130, 220], 1)
        self.assertEqual(self.train_images[1][245, 335], 149)

        # Test for SIMCEPImages_J15_C61_F29_s05_w2.TIF
        self.assertEqual(self.train_images[2][260, 360], 1)
        self.assertEqual(self.train_images[2][200, 350], 156)

if __name__ == "__main__":
    unittest.main()

