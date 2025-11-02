#!/usr/bin/env python3
"""
Simple test runner for ARC inference pipeline
Run specific test categories or all tests
"""

import sys
import unittest
import argparse


def run_specific_tests(test_category=None):
    """Run specific test category or all tests"""
    loader = unittest.TestLoader()

    if test_category:
        # Run specific test class
        test_module = __import__('test_arc_inference')
        test_class_name = f'Test{test_category}'

        if hasattr(test_module, test_class_name):
            suite = loader.loadTestsFromTestCase(getattr(test_module, test_class_name))
        else:
            print(f"❌ Test class '{test_class_name}' not found!")
            print("\nAvailable test categories:")
            print("  - GridEncoders")
            print("  - DataAugmentation")
            print("  - Prompting")
            print("  - InferencePipeline")
            print("  - Integration")
            return False
    else:
        # Run all tests
        suite = loader.discover('.', pattern='test_*.py')

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)

    return result.wasSuccessful()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run ARC inference tests')
    parser.add_argument(
        '--category',
        type=str,
        help='Test category to run (GridEncoders, DataAugmentation, Prompting, InferencePipeline, Integration)',
        default=None
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available test categories'
    )

    args = parser.parse_args()

    if args.list:
        print("Available test categories:")
        print("  - GridEncoders: Test grid encoding/decoding")
        print("  - DataAugmentation: Test data augmentation and transformations")
        print("  - Prompting: Test prompt generation and parsing")
        print("  - InferencePipeline: Test main inference components")
        print("  - Integration: Test end-to-end pipeline")
        return

    print("🧪 Running ARC Inference Tests...\n")

    success = run_specific_tests(args.category)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
