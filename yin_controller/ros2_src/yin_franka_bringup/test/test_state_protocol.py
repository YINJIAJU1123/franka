import math
import struct
import pytest
from yin_franka_bringup.state_protocol import PACKET, StatePacket, decode


def valid_packet():
    values = tuple(i / 100 for i in range(21)) + (0.4, 0.0, 0.3, 0., 0., 0., 1., 0.08) + tuple(i / 100 for i in range(14))
    return StatePacket(42, 4000, 3000, 2000, 1000, 5000, 15, values)


def test_wire_layout_and_roundtrip():
    packet = valid_packet()
    payload = packet.encode()
    assert len(payload) == 400
    assert payload[:8] == b"FRST\x02\x00\x0f\x00"
    assert struct.unpack_from("<Q", payload, 8) == (42,)
    assert decode(payload) == packet


def test_missing_state_is_explicit():
    packet = StatePacket(0, 123, 0, 0, 0, 0, 0, (float("nan"),) * 43)
    decoded = decode(packet.encode())
    assert decoded.flags == 0
    assert all(math.isnan(v) for v in decoded.values)


@pytest.mark.parametrize("mutate", [
    lambda b: b[:-1], lambda b: b + b"x", lambda b: b"OLD!" + b[4:],
    lambda b: b[:4] + b"\x03\x00" + b[6:],
    lambda b: b[:6] + b"\x08\x00" + b[8:],
])
def test_malformed_packets_rejected(mutate):
    with pytest.raises(ValueError):
        decode(mutate(valid_packet().encode()))


def test_nan_valid_data_rejected():
    fields = list(PACKET.unpack(valid_packet().encode()))
    fields[9] = float("nan")
    with pytest.raises(ValueError):
        decode(PACKET.pack(*fields))


def test_invalid_quaternion_rejected():
    fields = list(PACKET.unpack(valid_packet().encode()))
    fields[9 + 27] = 0.0
    with pytest.raises(ValueError):
        decode(PACKET.pack(*fields))
