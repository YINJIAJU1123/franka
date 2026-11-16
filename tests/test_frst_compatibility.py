"""Exercise the actual Yin encoder against both source decoders without ROS."""
import importlib.util
import math
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


YIN = load('test_yin_protocol', ROOT / 'yin_controller/ros2_src/yin_franka_bringup/yin_franka_bringup/state_protocol.py')
LIXIN = load('test_lixin_protocol', ROOT / 'state_receiver_source/deployment/frst_protocol.py')


def packet(flags=15):
    values = tuple(i / 100 for i in range(21)) + (.4, .0, .3, .0, .0, .0, 1., .08) + tuple(i / 100 for i in range(14))
    if flags == 0:
        values = (float('nan'),) * 43
    return YIN.StatePacket(42, 5000, 4000, 3000, 2000, 1000, flags, values)


class ProtocolCompatibility(unittest.TestCase):
    def test_sender_receiver_roundtrip(self):
        sent = packet()
        payload = sent.encode()
        self.assertEqual(len(payload), 400)
        self.assertEqual(payload[:8], b'FRST\x02\x00\x0f\x00')
        for decoded in [YIN.decode(payload), LIXIN.decode_frst(payload)]:
            self.assertEqual(decoded.sequence, sent.sequence)
            self.assertEqual(decoded.sent_ns, sent.sent_ns)
            self.assertEqual(decoded.gripper_ns, sent.gripper_ns)
            self.assertEqual(decoded.values, sent.values)

    def test_invalid_groups_are_explicit(self):
        payload = packet(0).encode()
        for decoder in [YIN.decode, LIXIN.decode_frst]:
            result = decoder(payload)
            self.assertEqual(result.flags, 0)
            self.assertTrue(all(math.isnan(x) for x in result.values))

    def test_corrupted_wire_data_is_rejected_by_both(self):
        payload = packet().encode()
        mutations = [payload[:-1], payload+b'x', b'OLD!'+payload[4:],
                     payload[:4]+struct.pack('<H', 3)+payload[6:],
                     payload[:6]+struct.pack('<H', 32)+payload[8:]]
        for mutated in mutations:
            for decoder in [YIN.decode, LIXIN.decode_frst]:
                with self.subTest(decoder=decoder.__name__, length=len(mutated)):
                    with self.assertRaises(ValueError):
                        decoder(mutated)

    def test_invalid_valid_group_values_are_rejected(self):
        for field, value in [(9, float('nan')), (9+27, 0.)]:
            fields = list(YIN.PACKET.unpack(packet().encode()))
            fields[field] = value
            payload = YIN.PACKET.pack(*fields)
            for decoder in [YIN.decode, LIXIN.decode_frst]:
                with self.assertRaises(ValueError):
                    decoder(payload)

    def test_negative_timestamps_are_rejected(self):
        fields = list(YIN.PACKET.unpack(packet().encode()))
        fields[4] = -1
        payload = YIN.PACKET.pack(*fields)
        for decoder in [YIN.decode, LIXIN.decode_frst]:
            with self.assertRaises(ValueError):
                decoder(payload)


if __name__ == '__main__':
    unittest.main()
