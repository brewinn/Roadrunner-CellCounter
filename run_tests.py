#!/usr/bin/python3
# This script simply runs all available tests.

if __name__ == "__main__":
    import unittest

    # For each file in the tests folder, add an import statement and an addTest statement
    from tests.test_import_tiff import TestImportTiff
    from tests.test_import_dataset import TestImportSyntheticDataset

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestImportTiff))
    suite.addTest(unittest.makeSuite(TestImportSyntheticDataset))

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print(result)
