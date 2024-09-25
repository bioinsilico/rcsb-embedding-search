import logging
import logging.config


def setup_logging():
    """Set up logging configuration."""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,  # Keep loggers from third-party libraries
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'INFO',
                'stream': 'ext://sys.stdout',  # Use sys.stdout
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'level': 'INFO',
                'filename': 'app.log',
                'encoding': 'utf8',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    }

    logging.config.dictConfig(logging_config)


setup_logging()
