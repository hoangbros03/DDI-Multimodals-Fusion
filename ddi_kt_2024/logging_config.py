import logging

def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Call the configure_logging function when the module is imported
configure_logging()