"""
Tests for RLTC encoder.
"""

import numpy as np
import pytest

from rltc import RLTCEncoder, parse_duration


class TestParseDuration:
    """Test duration string parsing."""

    def test_parse_milliseconds(self):
        assert parse_duration("100") == 100
        assert parse_duration("100ms") == 100
        assert parse_duration("5000ms") == 5000

    def test_parse_seconds(self):
        assert parse_duration("5s") == 5000
        assert parse_duration("30s") == 30000
        assert parse_duration("0.5s") == 500  # Will fail, fix needed

    def test_parse_minutes(self):
        assert parse_duration("1m") == 60000
        assert parse_duration("5m") == 300000
        assert parse_duration("90m") == 5400000

    def test_parse_hours(self):
        assert parse_duration("1h") == 3600000
        assert parse_duration("2h") == 7200000
        assert parse_duration("24h") == 86400000

    def test_parse_colon_format(self):
        assert parse_duration("30") == 30000  # Treated as seconds
        assert parse_duration("1:30") == 90000  # 1 min 30 sec
        assert parse_duration("1:05:30") == 3810000  # 1 hr 5 min 30 sec


class TestRLTCEncoder:
    """Test FSK encoder."""

    def test_encoder_init(self):
        encoder = RLTCEncoder()
        assert encoder.sample_rate == 44100
        assert encoder.mark_freq == 2200
        assert encoder.space_freq == 1200
        assert encoder.baud_rate == 1200

    def test_encoder_custom_params(self):
        encoder = RLTCEncoder(
            sample_rate=48000,
            mark_freq=2400,
            space_freq=1400,
            baud_rate=2400,
        )
        assert encoder.sample_rate == 48000
        assert encoder.mark_freq == 2400
        assert encoder.space_freq == 1400
        assert encoder.baud_rate == 2400

    def test_generate_bit(self):
        encoder = RLTCEncoder()

        # Generate a mark bit (1)
        mark_samples = encoder._generate_bit(1)
        assert len(mark_samples) == int(44100 / 1200)  # samples per bit

        # Generate a space bit (0)
        space_samples = encoder._generate_bit(0)
        assert len(space_samples) == int(44100 / 1200)

        # Samples should be in range [-1, 1]
        assert np.all(np.abs(mark_samples) <= 1.0)
        assert np.all(np.abs(space_samples) <= 1.0)

    def test_continuous_phase(self):
        """Test that phase is continuous between bits."""
        encoder = RLTCEncoder()

        # Generate multiple bits
        all_samples = []
        for bit in [1, 0, 1, 1, 0]:
            samples = encoder._generate_bit(bit)
            all_samples.extend(samples)

        # Check for discontinuities (large jumps between consecutive samples)
        arr = np.array(all_samples)
        diffs = np.abs(np.diff(arr))
        # Allow some difference but not huge jumps (would indicate phase reset)
        # Max expected diff is approximately 2 * sin(omega * dt) for worst case
        max_diff = np.max(diffs)
        assert max_diff < 0.5, f"Large discontinuity detected: {max_diff}"

    def test_generate_short_duration(self):
        """Test generating a very short countdown."""
        encoder = RLTCEncoder()

        # Generate 100ms countdown (very short)
        samples, sr = encoder.generate(duration_ms=100, packet_rate_hz=30)

        assert sr == 44100
        expected_samples = int(44100 * 0.1)
        assert len(samples) == expected_samples

    def test_generate_amplitude(self):
        """Test amplitude scaling."""
        encoder = RLTCEncoder()

        # Generate with different amplitudes
        samples_half, _ = encoder.generate(duration_ms=100, amplitude=0.5)
        samples_full, _ = encoder.generate(duration_ms=100, amplitude=1.0)

        # Full amplitude should have higher RMS
        rms_half = np.sqrt(np.mean(samples_half ** 2))
        rms_full = np.sqrt(np.mean(samples_full ** 2))

        assert rms_full > rms_half
        assert abs(rms_half * 2 - rms_full) < 0.1  # Approx 2x difference

    def test_generate_five_second(self):
        """Test generating a 5-second countdown."""
        encoder = RLTCEncoder()

        samples, sr = encoder.generate(
            duration_ms=5000,
            packet_rate_hz=30,
        )

        expected_samples = int(44100 * 5)
        assert len(samples) == expected_samples

        # Check that samples are within valid range
        assert np.all(np.abs(samples) <= 1.0)
