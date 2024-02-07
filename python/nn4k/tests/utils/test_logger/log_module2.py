import unittest

from nn4k.utils.logger import logger


def get_logger_from_log_module2():
    return logger


class TestLogger(unittest.TestCase):
    def testSingletonDefaultLogger(self):
        from utils.test_logger.log_module1 import get_logger_from_log_module1
        from nn4k.utils.logger import DEFAULT_LOGGER_KEY

        logger1 = get_logger_from_log_module1()
        logger2 = get_logger_from_log_module2()

        self.assertEqual(
            logger1.name,
            DEFAULT_LOGGER_KEY,
            f"default logger name should be {DEFAULT_LOGGER_KEY}",
        )

        self.assertEqual(
            id(logger1),
            id(logger2),
            "there should be only one global default logger (singleton)",
        )


if __name__ == "__main__":
    unittest.main()
