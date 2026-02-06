"""
RLTC Decoder - Real-time decodes FSK timecode from audio input.
"""

import array
import queue
import threading
import time
from collections import deque
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

from . import SAMPLE_RATE, MARK_FREQ, SPACE_FREQ, BAUD_RATE
from .packet import Packet, SYNC_PATTERN


class FSKDemodulator:
    """
    FSK demodulator using correlation-based detection.

    More robust than simple zero-crossing detection for noisy signals.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        mark_freq: int = MARK_FREQ,
        space_freq: int = SPACE_FREQ,
        baud_rate: int = BAUD_RATE,
    ):
        """
        Initialize demodulator.

        Args:
            sample_rate: Audio sample rate (Hz)
            mark_freq: Frequency for logic 1 (Hz)
            space_freq: Frequency for logic 0 (Hz)
            baud_rate: Bit rate (bits per second)
        """
        self.sample_rate = sample_rate
        self.mark_freq = mark_freq
        self.space_freq = space_freq
        self.baud_rate = baud_rate

        self.samples_per_bit = int(sample_rate / baud_rate)

        # Generate correlation templates for one bit period
        self.mark_template = self._generate_template(mark_freq)
        self.space_template = self._generate_template(space_freq)

        # Buffer for incoming samples
        self.buffer = np.zeros(0)

        # Phase tracking for bit synchronization
        self.bit_phase = 0.0

    def _generate_template(self, freq: int) -> np.ndarray:
        """Generate sine wave template for correlation."""
        duration = self.samples_per_bit / self.sample_rate
        t = np.linspace(0, duration, self.samples_per_bit, endpoint=False)
        return np.sin(2 * np.pi * freq * t)

    def _correlate(self, samples: np.ndarray) -> float:
        """
        Correlate samples against mark/space templates.

        Returns:
            float: Positive for mark, negative for space, magnitude indicates confidence
        """
        if len(samples) < self.samples_per_bit:
            return 0.0

        # Take exactly one bit period
        bit_samples = samples[:self.samples_per_bit]

        # Normalize to prevent amplitude issues
        rms = np.sqrt(np.mean(bit_samples ** 2))
        if rms < 0.01:  # Silence/near silence
            return 0.0

        bit_samples = bit_samples / rms

        # Correlate with both templates
        mark_corr = np.dot(bit_samples, self.mark_template)
        space_corr = np.dot(bit_samples, self.space_template)

        # Return difference (positive = mark, negative = space)
        return mark_corr - space_corr

    def reset(self):
        """Reset demodulator state."""
        self.buffer = np.zeros(0)
        self.bit_phase = 0.0

    def process(self, samples: np.ndarray) -> list[int]:
        """
        Process incoming audio samples and extract bits.

        Args:
            samples: Input audio samples (float32, -1.0 to 1.0)

        Returns:
            List of decoded bits (0 or 1)
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, samples])

        bits = []

        # Process as many bits as possible
        while len(self.buffer) >= self.samples_per_bit:
            # Extract bit period
            bit_samples = self.buffer[:self.samples_per_bit]

            # Correlate to determine bit value
            correlation = self._correlate(bit_samples)

            # Hysteresis for noise immunity
            if correlation > 0.3:
                bits.append(1)
            elif correlation < -0.3:
                bits.append(0)
            else:
                # Ambiguous - skip this bit
                pass

            # Remove processed samples
            # Use fractional bit phase for better synchronization
            samples_to_remove = int(self.samples_per_bit)
            self.buffer = self.buffer[samples_to_remove:]

        return bits


class PacketDetector:
    """
    Detects and validates RLTC packets from a bit stream.

    Uses correlation to find the preamble + sync pattern.
    """

    def __init__(self):
        """Initialize packet detector."""
        # Expected bit pattern for detection
        # Preamble (32 alternating) + Sync (16)
        preamble_bits = Packet.get_preamble_bits()
        sync_bits = Packet.get_sync_bits()
        self.detection_pattern = preamble_bits + sync_bits

        # Accumulated bits
        self.bit_buffer: deque[int] = deque(maxlen=256)

        # Packet bit queue (for actual packet extraction)
        self.packet_bits: list[int] = []

        # State
        self.in_packet = False
        self.bits_needed = 0

    def reset(self):
        """Reset detector state."""
        self.bit_buffer.clear()
        self.packet_bits = []
        self.in_packet = False
        self.bits_needed = 0

    def _find_pattern(self) -> Optional[int]:
        """
        Find detection pattern in bit buffer.

        Returns:
            Index after pattern if found, None otherwise
        """
        if len(self.bit_buffer) < len(self.detection_pattern):
            return None

        # Convert to array for easier processing
        bits = list(self.bit_buffer)

        # Simple correlation-based search
        # Look for the alternating preamble first
        for i in range(len(bits) - len(self.detection_pattern)):
            match = True
            for j, expected in enumerate(self.detection_pattern):
                if bits[i + j] != expected:
                    match = False
                    break

            if match:
                # Found pattern - return index after it
                return i + len(self.detection_pattern)

        return None

    def feed_bits(self, bits: list[int]) -> list[Packet]:
        """
        Feed bits to detector and return any complete packets.

        Args:
            bits: List of bits (0 or 1)

        Returns:
            List of valid packets found
        """
        packets = []

        for bit in bits:
            self.bit_buffer.append(bit)

            if self.in_packet:
                # Collecting packet bits
                self.packet_bits.append(bit)
                self.bits_needed -= 1

                if self.bits_needed <= 0:
                    # Packet complete - try to decode
                    packet = self._decode_packet()
                    if packet:
                        packets.append(packet)
                    self.in_packet = False
                    self.packet_bits = []
            else:
                # Looking for pattern
                pattern_end = self._find_pattern()
                if pattern_end is not None:
                    # Found sync - start collecting packet
                    # After pattern, we need: time (32) + counter (16) + crc (16) = 64 bits = 8 bytes
                    self.in_packet = True
                    self.bits_needed = 64  # 8 bytes * 8 bits
                    self.packet_bits = []

                    # Remove bits up to pattern end
                    for _ in range(pattern_end):
                        if self.bit_buffer:
                            self.bit_buffer.popleft()

        return packets

    def _decode_packet(self) -> Optional[Packet]:
        """
        Decode collected packet bits.

        Returns:
            Packet if valid, None otherwise
        """
        if len(self.packet_bits) < 64:
            return None

        # Convert bits to bytes
        data = bytearray()
        for i in range(0, 64, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.packet_bits[i + j]
            data.append(byte)

        # Decode (sync + time + counter + crc)
        # Reconstruct full 12-byte packet (add back sync)
        full_packet = bytes([0x16, 0xA3]) + bytes(data)  # Sync + payload

        try:
            packet = Packet.decode(full_packet)
            return packet
        except Exception:
            return None


class RLTCDecoder:
    """
    Real-time RLTC decoder from audio input.

    Outputs countdown time with smoothing for robustness.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        callback: Optional[Callable[[int], None]] = None,
        device: Optional[int] = None,
    ):
        """
        Initialize decoder.

        Args:
            sample_rate: Audio sample rate (Hz)
            callback: Optional callback for each valid time reading (time_ms)
            device: Audio input device (None = default)
        """
        self.sample_rate = sample_rate
        self.callback = callback

        # Components
        self.demodulator = FSKDemodulator(sample_rate)
        self.detector = PacketDetector()

        # State tracking
        self.last_packet_counter: Optional[int] = None
        self.last_time_ms: Optional[int] = None

        # Smoothing - use recent valid readings
        self.time_history: deque[int] = deque(maxlen=5)

        # Audio stream
        self.stream: Optional[sd.InputStream] = None
        self.device = device

        # Statistics
        self.packets_received = 0
        self.packets_invalid = 0
        self.last_update_time: Optional[float] = None

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        """
        Called by sounddevice for each audio block.

        Demodulates and extracts packets from incoming audio.
        """
        if status:
            print(f"Audio status: {status}")

        # Convert to float and flatten
        samples = indata.astype(np.float32).flatten()

        # Demodulate to bits
        bits = self.demodulator.process(samples)

        if bits:
            # Feed to packet detector
            packets = self.detector.feed_bits(bits)

            for packet in packets:
                self._handle_packet(packet)

    def _handle_packet(self, packet: Packet):
        """
        Handle a received packet.

        Args:
            packet: Valid packet received
        """
        self.packets_received += 1

        # Check for missing packets (counter jumped)
        if self.last_packet_counter is not None:
            expected = (self.last_packet_counter + 1) & 0xFFFF
            if packet.packet_counter != expected:
                # Packets missing - but still use this one
                pass

        self.last_packet_counter = packet.packet_counter

        # Add to history for smoothing
        self.time_history.append(packet.time_remaining_ms)

        # Get median of recent readings (reduces impact of outliers)
        if len(self.time_history) >= 3:
            time_ms = int(np.median(list(self.time_history)))
        else:
            time_ms = packet.time_remaining_ms

        # Check for consistency - time should decrease
        if self.last_time_ms is not None:
            # Allow some increase due to jitter, but not large jumps
            if time_ms > self.last_time_ms + 5000:
                # Huge jump backward - probably error, ignore
                return

        self.last_time_ms = time_ms
        self.last_update_time = time.time()

        # Call user callback if provided
        if self.callback:
            self.callback(time_ms)

    def start(self):
        """Start decoding from audio input."""
        if self.stream is not None:
            return  # Already running

        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=0,  # Use default
        )

        self.stream.start()

    def stop(self):
        """Stop decoding."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_time_remaining(self) -> Optional[int]:
        """
        Get the current time remaining.

        Returns:
            Time in milliseconds, or None if no valid data received recently
        """
        # Consider data stale if no updates for 500ms
        if self.last_update_time is None:
            return None

        if time.time() - self.last_update_time > 0.5:
            return None

        return self.last_time_ms

    def get_statistics(self) -> dict:
        """
        Get decoder statistics.

        Returns:
            Dict with: packets_received, packets_invalid, time_remaining
        """
        return {
            "packets_received": self.packets_received,
            "packets_invalid": self.packets_invalid,
            "time_remaining": self.get_time_remaining(),
        }


def decode_file(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[float, int]]:
    """
    Decode RLTC from an audio file (for testing).

    Args:
        file_path: Path to audio file
        sample_rate: Expected sample rate

    Returns:
        List of (time_seconds, time_remaining_ms) tuples
    """
    import soundfile as sf

    samples, sr = sf.read(file_path)

    # Resample if needed
    if sr != sample_rate:
        from scipy import signal
        num_samples = int(len(samples) * sample_rate / sr)
        samples = signal.resample(samples, num_samples)

    demodulator = FSKDemodulator(sample_rate)
    detector = PacketDetector()

    results = []
    sample_position = 0
    samples_per_block = sample_rate  # Process 1 second at a time

    while sample_position < len(samples):
        block = samples[sample_position:sample_position + samples_per_block]
        bits = demodulator.process(block)
        packets = detector.feed_bits(bits)

        current_time = sample_position / sample_rate

        for packet in packets:
            results.append((current_time, packet.time_remaining_ms))

        sample_position += samples_per_block

    return results
