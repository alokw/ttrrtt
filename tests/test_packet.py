"""
Tests for RLTC packet encoding/decoding.
"""

import pytest

from rltc import Packet, CRC16CCITT


class TestCRC:
    """Test CRC implementation."""

    def test_crc_compute(self):
        """Test basic CRC computation."""
        # Test vector: "123456789" -> CRC-16-CCITT = 0x29B1
        data = b"\x31\x32\x33\x34\x35\x36\x37\x38\x39"
        crc = CRC16CCITT.compute(data)
        assert crc == 0x29B1

    def test_crc_verify(self):
        """Test CRC verification."""
        data = b"test data"
        crc = CRC16CCITT.compute(data)
        assert CRC16CCITT.verify(data, crc) is True
        assert CRC16CCITT.verify(data, crc ^ 0xFFFF) is False


class TestPacket:
    """Test packet encoding and decoding."""

    def test_packet_round_trip(self):
        """Test encoding and decoding a packet."""
        original = Packet(
            time_remaining_ms=5000,
            packet_counter=42,
        )

        encoded = original.encode()
        decoded = Packet.decode(encoded)

        assert decoded is not None
        assert decoded.time_remaining_ms == 5000
        assert decoded.packet_counter == 42

    def test_packet_with_crc_error(self):
        """Test that corrupted CRC is detected."""
        original = Packet(
            time_remaining_ms=10000,
            packet_counter=1,
        )

        encoded = bytearray(original.encode())
        # Corrupt the data
        encoded[5] ^= 0xFF

        decoded = Packet.decode(bytes(encoded))
        assert decoded is None  # Should fail CRC

    def test_packet_wrong_sync(self):
        """Test that wrong sync word is rejected."""
        original = Packet(
            time_remaining_ms=10000,
            packet_counter=1,
        )

        encoded = bytearray(original.encode())
        # Corrupt sync word
        encoded[0] ^= 0xFF

        decoded = Packet.decode(bytes(encoded))
        assert decoded is None

    def test_packet_max_values(self):
        """Test packet with maximum values."""
        original = Packet(
            time_remaining_ms=0xFFFFFFFF,  # Max 32-bit
            packet_counter=0xFFFF,  # Max 16-bit
        )

        encoded = original.encode()
        decoded = Packet.decode(encoded)

        assert decoded.time_remaining_ms == 0xFFFFFFFF
        assert decoded.packet_counter == 0xFFFF

    def test_packet_zero(self):
        """Test packet with zero values."""
        original = Packet(
            time_remaining_ms=0,
            packet_counter=0,
        )

        encoded = original.encode()
        decoded = Packet.decode(encoded)

        assert decoded.time_remaining_ms == 0
        assert decoded.packet_counter == 0

    def test_packet_encode_length(self):
        """Test that encoded packet has correct length."""
        packet = Packet(5000, 1)
        encoded = packet.encode()
        assert len(encoded) == 10  # sync(2) + time(4) + counter(2) + crc(2)

    def test_packet_with_preamble_length(self):
        """Test that packet with preamble has correct length."""
        packet = Packet(5000, 1)
        encoded = packet.encode_with_preamble()
        assert len(encoded) == 14  # preamble(4) + packet(10)

    def test_preamble_pattern(self):
        """Test that preamble is correct alternating pattern."""
        packet = Packet(5000, 1)
        encoded = packet.encode_with_preamble()

        # First 4 bytes should be 0xAA 0xAA 0xAA 0xAA
        assert encoded[0] == 0xAA
        assert encoded[1] == 0xAA
        assert encoded[2] == 0xAA
        assert encoded[3] == 0xAA

    def test_packet_counter_limits(self):
        """Test packet counter validation."""
        # Valid
        Packet(5000, 0)
        Packet(5000, 0xFFFF)

        # Invalid
        with pytest.raises(ValueError):
            Packet(5000, -1)
        with pytest.raises(ValueError):
            Packet(5000, 0x10000)

    def test_time_remaining_limits(self):
        """Test time remaining validation."""
        # Valid
        Packet(0, 1)
        Packet(0xFFFFFFFF, 1)

        # Invalid
        with pytest.raises(ValueError):
            Packet(-1, 1)
        with pytest.raises(ValueError):
            Packet(0x100000000, 1)
