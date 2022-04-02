# Test that the import_dataset script works as intended

import unittest

# For finding test dataset
import os

# Import the relevant function after installing the package
from cell_counter.import_dataset import load_synthetic_dataset


class TestImportSyntheticDataset(unittest.TestCase):
    def setUp(self):
        # Path to test dataset
        tests_dir = os.path.dirname(__file__)
        path_to_images = tests_dir + "/../resources/test_dataset/"
        # Import dataset
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = load_synthetic_dataset(is_random=False, num=20, path=path_to_images)

    def test_image_sizes(self):
        self.assertEqual(len(self.train_images), 16)
        self.assertEqual(len(self.test_images), 4)

    def test_label_sizes(self):
        self.assertEqual(len(self.train_labels), 16)
        self.assertEqual(len(self.test_labels), 4)

    def test_bad_size(self):
        self.assertRaises(Exception, load_synthetic_dataset, **{"num": -1})
        self.assertRaises(Exception, load_synthetic_dataset, **{"num": 100000000000})

    def test_imported_images(self):
        # With the is_random set to False, the first three images in the
        # training set should be SIMCEPImages_A15_C61_F1_s14_w1.TIF,
        # SIMCEPImages_B18_C74_F4_s04_w2.TIF, and
        # SIMCEPImages_B18_C74_F4_s10_w2.TIF



        # Test for SIMCEPImages_A15_C61_F1_s14_w1.TIF
        self.assertEqual(self.train_images[0][170, 350], 1)
        self.assertEqual(self.train_images[0][170, 400], 111)

        # Test for SIMCEPImages_B18_C74_F4_s04_w2.TIF
        self.assertEqual(self.train_images[1][130, 220], 1)
        self.assertEqual(self.train_images[1][245, 335], 149)

        # Test for SIMCEPImages_B18_C74_F4_s10_w2.TIF
        self.assertEqual(self.train_images[2][260, 360], 1)
        self.assertEqual(self.train_images[2][200, 350], 156)

    def test_imported_labels(self):
        # Similar to the  import_images test, we check that the first few
        #  imported labels are correct
        self.assertEqual(self.train_labels[0], 61)
        self.assertEqual(self.train_labels[1], 74)
        self.assertEqual(self.train_labels[2], 74)


if __name__ == "__main__":
    unittest.main()
