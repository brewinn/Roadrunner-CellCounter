# Test that the cnn_cellcounter script works as intended

import unittest

# For finding test dataset
import os

# Import the relevant functions after installing the package
from cell_counter.cnn_cellcounter import (
    build_cnn,
    compile_cnn,
    cnn_preprocess_data,
    run_cnn,
)
from cell_counter.import_dataset import load_synthetic_dataset


class TestCNN(unittest.TestCase):
    def test_cnn_preprocess_data(self):
        test_dir = os.path.dirname(__file__)
        test_dataset_path = test_dir + "/../resources/test_dataset/"
        dataset = cnn_preprocess_data(path=test_dataset_path, num=25)

        # Check resolution
        self.assertEqual(dataset[0][0][0].shape, (128, 128))
        self.assertEqual(dataset[1][0][0].shape, (128, 128))

        # Check range
        image = dataset[0][0][0]
        above = image <= 1
        below = image >= 0
        self.assertTrue(above.all())
        self.assertTrue(below.all())

    def test_cnn_model_build(self):
        # Simply check that the model builds and compiles without error
        model = build_cnn()
        compile_cnn(model)

    def test_cnn_run(self):
        # Check that the model runs on the test set without failure
        test_dir = os.path.dirname(__file__)
        test_dataset_path = test_dir + "/../resources/test_dataset/"
        model = build_cnn()
        compile_cnn(model)
        run_cnn(
            model,
            path=test_dataset_path,
            image_number=25,
            checkpointing=False,
            verbose=0,
        )


if __name__ == "__main__":
    unittest.main()
