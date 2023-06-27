from rclpy.serialization import serialize_message
import hashlib, copy

def getMessageHash(msg):
	"""
	Returns the MD5 hash for a message.
	
	:param msg: The message to calculate a hash for. Must have a ugrdv_msgs/Meta field named meta.
	:returns: A hex string containing the hash.
	"""
	serialized = serialize_message(msg)
	hash = hashlib.md5(serialized, usedforsecurity=False)
	return hash.hexdigest().upper()

def getMessageHashMeta(msg):
    """
    Returns the MD5 hash for a message with a meta field.

    :param msg: The message to calculate a hash for. Must have a ugrdv_msgs/Meta field named meta.
    :returns: A hex string containing the hash.
    """
    msg = copy.deepcopy(msg)
    msg.meta.hash = ""
    hash = getMessageHash(msg)
    return hash

def rosTimestampToMillis(stamp):
    return int(stamp.sec * 1e3 + stamp.nanosec * 1e-6)
