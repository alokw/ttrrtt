"""
RLTC - Reverse Linear Timecode
A robust FSK-based countdown timecode system.
"""

__version__ = "0.1.0"

# Protocol constants
MARK_FREQ = 2200  # Hz - logic 1
SPACE_FREQ = 1200  # Hz - logic 0
BAUD_RATE = 1200  # bits per second
SAMPLE_RATE = 44100  # Hz (default)
PACKET_RATE = 30  # packets per second (default)

# Packet structure
PREAMBLE_BITS = 32  # 0xAAAAAAAA pattern
SYNC_WORD = 0x16A3  # 16-bit sync
TIME_REMAINING_BITS = 32  # milliseconds
PACKET_COUNTER_BITS = 16  # increments each packet
CRC_BITS = 16  # CRC-16-CCITT

TOTAL_BITS = (
    PREAMBLE_BITS +
    16 +  # sync word
    TIME_REMAINING_BITS +
    PACKET_COUNTER_BITS +
    CRC_BITS
)  # = 112 bits

# Sync pattern (binary 0001011010100011)
# Chosen for good autocorrelation properties
SYNC_PATTERN = [
    0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1
]

from .packet import Packet, CRC16CCITT
from .encoder import RLTCEncoder, parse_duration
from .decoder import RLTCDecoder, decode_file

__all__ = [
    "Packet",
    "CRC16CCITT",
    "RLTCEncoder",
    "RLTCDecoder",
    "decode_file",
    "parse_duration",
]
