#!/usr/bin/python3
# This script simply runs all available tests.

if __name__ == "__main__":
    import unittest

    # For each file in the tests folder, add an import statement and an addTest statement
    from tests.test_import_tiff import TestImportTiff
    from tests.test_import_dataset import TestImportSyntheticDataset
    from tests.test_cnn_cellcounter import TestCNN

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestImportTiff))
    suite.addTest(unittest.makeSuite(TestImportSyntheticDataset))
    # CNN tests run on old data, may update later
    #suite.addTest(unittest.makeSuite(TestCNN))

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # The script should exit with failure if some test did not pass.
    # This allows CI to report if some test failed, rather than having to look
    # through the logs.
    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
