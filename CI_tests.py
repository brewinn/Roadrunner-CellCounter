# This script runs all the tests that do not need access to the dataset.
# This allows us to use github actions to automatically check the code when it
#  is pushed to the repo.

if __name__ == "__main__":
    import unittest

    # For each file in the tests folder, add an import statement and an addTest statement
    from tests.test_import_tiff import TestImportTiff

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestImportTiff))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
