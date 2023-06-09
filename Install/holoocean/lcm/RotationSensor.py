"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class RotationSensor(object):
    __slots__ = ["timestamp", "roll", "pitch", "yaw"]

    __typenames__ = ["int64_t", "float", "float", "float"]

    __dimensions__ = [None, None, None, None]

    def __init__(self):
        self.timestamp = 0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(RotationSensor._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qfff", self.timestamp, self.roll, self.pitch, self.yaw))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != RotationSensor._get_packed_fingerprint():
            raise ValueError("Decode error")
        return RotationSensor._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = RotationSensor()
        self.timestamp, self.roll, self.pitch, self.yaw = struct.unpack(">qfff", buf.read(20))
        return self
    _decode_one = staticmethod(_decode_one)

    def _get_hash_recursive(parents):
        if RotationSensor in parents: return 0
        tmphash = (0xa57d6ed5ba175b17) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if RotationSensor._packed_fingerprint is None:
            RotationSensor._packed_fingerprint = struct.pack(">Q", RotationSensor._get_hash_recursive([]))
        return RotationSensor._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", RotationSensor._get_packed_fingerprint())[0]

