from orm import logger

log = logger.get_logger("test", level="debug")

log.debug("Đây là debug!")
log.info("Đây là info!")
log.warning("Đây là warning!")
log.error("Đây là error!")
log.critical("Đây là critical!")