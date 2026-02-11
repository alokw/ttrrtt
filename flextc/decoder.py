"""
SMPTE/LTC Decoder with Countdown Support

Decodes SMPTE/LTC audio and detects whether it's counting up or down.
Supports both standard SMPTE timecode and FlexTC countdown mode.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from flextc.smpte_packet import Timecode, FrameRate
from flextc.biphase import BiphaseMDecoder

# Module-level logger
_logger = logging.getLogger(__name__)


class OSCBroadcaster:
    """
    OSC broadcaster for sending timecode data.

    Broadcasts timecode as a string in the format "HH:MM:SS:FF" or "HH:MM:SS;FF" (for drop-frame).
    Uses the path /flextc/ltc or /flextc/count based on direction.
    """

    def __init__(self, address: str = "255.255.255.255", port: int = 9988):
        """
        Initialize OSC broadcaster.

        Args:
            address: IP address or hostname for OSC broadcast (default: 255.255.255.255)
            port: UDP port for OSC (default: 9988)
        """
        self.address = address
        self.port = port
        self._client = None
        self._enabled = False

    def enable(self):
        """Enable OSC broadcasting."""
        try:
            from pythonosc import udp_client
            self._client = udp_client.UDPClient(self.address, self.port)
            self._enabled = True
            _logger.info(f"OSC broadcasting enabled to {self.address}:{self.port}")
        except ImportError:
            _logger.warning("python-osc not installed. Install with: pip install python-osc")
            self._enabled = False
        except Exception as e:
            _logger.error(f"Failed to initialize OSC client: {e}")
            self._enabled = False

    def disable(self):
        """Disable OSC broadcasting and close connection."""
        self._enabled = False
        if self._client is not None:
            try:
                # python-osc UDPClient doesn't have an explicit close method
                # Just clear the reference
                self._client = None
            except Exception:
                pass

    def send_timecode(self, tc: Timecode):
        """
        Send timecode via OSC.

        Args:
            tc: Timecode object to broadcast
        """
        if not self._enabled or self._client is None:
            return

        try:
            # Format timecode as string (HH:MM:SS:FF or HH:MM:SS;FF for drop-frame)
            separator = ";" if tc.is_drop_frame else ":"
            tc_string = f"{tc.hours:02d}:{tc.minutes:02d}:{tc.seconds:02d}{separator}{tc.frames:02d}"

            # Determine OSC path based on direction
            path = "/flextc/ltc" if tc.count_up else "/flextc/count"

            # Send OSC message
            from pythonosc import udp_client
            # Build and send OSC message
            builder = udp_client.OscMessageBuilder(address=path)
            builder.add_arg(tc_string)
            msg = builder.build()
            self._client.send(msg)

        except Exception as e:
            # Don't spam logs - only log once per error type ideally
            # For now, just log on error
            _logger.debug(f"OSC send failed: {e}")


class Decoder:
    """
    SMPTE/LTC decoder with countdown detection.

    Decodes audio frames and extracts timecode information.
    Automatically detects count-up vs countdown mode.
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        frame_rate: float = 30.0,
        callback: Optional[callable] = None,
        device: Optional[int] = None,
        channel: int = 0,
        debug: bool = False,
        osc_enabled: bool = False,
        osc_address: str = "127.0.0.1",
        osc_port: int = 9988,
    ):
        """
        Initialize decoder.

        Args:
            sample_rate: Audio sample rate (Hz) - None to auto-detect from device
            frame_rate: Frame rate (fps) - None for auto-detect
            callback: Optional callback for each frame (timecode: Timecode)
            device: Audio input device (None = default)
            channel: Audio channel to listen to (0 = first/left, 1 = second/right)
            debug: Enable debug logging
            osc_enabled: Enable OSC re-distribution
            osc_address: OSC address (default: 127.0.0.1)
            osc_port: OSC UDP port (default: 9988)
        """
        # Auto-detect sample rate from device if not specified
        if sample_rate is None:
            sample_rate = self._get_device_sample_rate(device)

        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.callback = callback
        self.channel = channel
        self.debug = debug

        # OSC re-distribution
        self.osc_broadcaster: Optional[OSCBroadcaster] = None
        if osc_enabled:
            self.osc_broadcaster = OSCBroadcaster(address=osc_address, port=osc_port)
            self.osc_broadcaster.enable()

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            _logger.setLevel(logging.DEBUG)
            _logger.debug(f"Decoder initialized: sample_rate={sample_rate}, frame_rate={frame_rate}, channel={channel}, osc={osc_enabled}")

        # Biphase decoder - created when frame rate is known
        self.biphase: Optional[BiphaseMDecoder] = None

        # Buffer for auto-detection (used when frame_rate is None or signal is lost)
        self._detection_buffer: List[np.ndarray] = []
        self._detection_buffer_samples = 0
        self._detection_threshold = sample_rate // 10  # 0.1 seconds of audio for detection
        self._signal_loss_threshold = sample_rate * 0.5  # 500ms without data = signal lost (less aggressive)
        self._last_valid_sample_time: Optional[float] = None
        self._in_detection_mode = (frame_rate is None)
        self._last_signal_was_strong = False  # Track if we had strong signal in previous callback

        if frame_rate is not None:
            self.biphase = BiphaseMDecoder(sample_rate, frame_rate)
            self._last_valid_sample_time = time.time()

        # State tracking
        self.current_timecode: Optional[Timecode] = None
        self.packets_received = 0
        self._last_timecode_value = None  # For detecting stuck frames
        self._stuck_frame_count = 0  # Consecutive repeats of same timecode
        self._last_valid_hour: Optional[int] = None  # For hour continuity checking
        self._hour_confidence: int = 0  # Number of consecutive frames with same hour range

        # Freewheeling: continue incrementing from last known timecode when glitches occur
        self._freewheel_base_timecode: Optional[Timecode] = None  # Last known good timecode
        self._freewheel_base_time: Optional[float] = None  # When that timecode was received
        self._freewheel_timeout = 0.25  # seconds - how long to freewheel before accepting jump
        self._is_freewheeling = False  # Currently in freewheel mode?

        # Audio stream
        self.stream: Optional[sd.InputStream] = None
        self.device = device

        # Last update time
        self.last_update_time: Optional[float] = None

    def _get_device_sample_rate(self, device: Optional[int]) -> int:
        """
        Get the default sample rate for an audio device.

        Args:
            device: Device index (None for default device)

        Returns:
            Default sample rate for the device
        """
        try:
            device_info = sd.query_devices(device)
            sr = device_info['default_samplerate']
            _logger.info(f"Auto-detected sample rate: {sr} Hz from device: {device_info['name']}")
            return int(sr)
        except Exception as e:
            _logger.warning(f"Could not auto-detect sample rate, using default 48000 Hz: {e}")
            return 48000

    def _is_hour_plausible(self, hours: int) -> bool:
        """
        Check if an hour value is plausible given the history.

        At high hour values (100+), bit errors in the upper bits can cause
        large jumps. This check rejects obviously invalid hour values.

        IMPORTANT: This should only reject CLEARLY impossible values.
        Legitimate timecode jumps (like seeking) should always be allowed.

        Args:
            hours: The hour value to check

        Returns:
            True if the hour is plausible, False if it's likely a bit error
        """
        if self._last_valid_hour is None:
            # No history, accept any valid value
            return True

        # Calculate the difference
        hour_diff = abs(hours - self._last_valid_hour)

        # With 4-bit tens encoding (max 159), max single-bit error is +/- 60 hours
        # A jump of more than 80 hours is definitely a bit error
        if hour_diff > 80:
            if self.debug:
                _logger.debug(f"Hour jump > 80 (likely bit error): {self._last_valid_hour} -> {hours}")
            return False

        # For high hours (80+), check for specific bit error patterns
        # With 4-bit tens: bits 54-55 store upper 2 bits, bits 56-57 store lower 2 bits
        # Single bit flips in upper 2 bits cause jumps of 40, 60, or combinations
        if self._last_valid_hour >= 80:
            # Check for specific bit error patterns in 4-bit tens encoding
            # - Bit 54 flip (bit 2 of tens): +/- 40 hours
            # - Bit 55 flip (bit 3 of tens): +/- 60 hours
            # - Bit 56 flip (bit 0 of tens): +/- 10 hours
            # - Bit 57 flip (bit 1 of tens): +/- 20 hours
            bit_error_patterns = (10, 20, 30, 40, 50, 60, 70)
            if hour_diff in bit_error_patterns:
                # Check if units digit is the same (suggests only tens bits affected)
                if hours % 10 == self._last_valid_hour % 10:
                    if self.debug:
                        _logger.debug(f"Hour jump with same units (likely bit error): {self._last_valid_hour} -> {hours} (diff: {hour_diff})")
                    return False

        return True

    def _is_timecode_jump_reasonable(self, tc: 'Timecode', current_time: float) -> tuple[bool, Optional[Timecode]]:
        """
        Check if a timecode jump is reasonable or if we should freewheel.

        Freewheeling strategy:
        - When a bad jump is detected, return (False, freewheeled_timecode)
        - The freewheeled timecode continues incrementing from the last known good value
        - If incoming timecode matches expected freewheled value, exit freewheel mode
        - If freewheel timeout expires, accept the jump and reset

        Args:
            tc: The new timecode to check
            current_time: Current timestamp

        Returns:
            Tuple of (should_accept: bool, timecode_to_use: Optional[Timecode])
            - If should_accept is True, use tc
            - If should_accept is False, use the returned freewheeled timecode
        """
        # Get frame rate
        fps = self.frame_rate if self.frame_rate else (tc.fps if tc.fps else 30.0)

        # No previous reference - accept anything, don't freewheel
        if self._freewheel_base_timecode is None:
            self._is_freewheeling = False
            return True, None

        # If we're not currently freewheeling, check if this jump is suspicious
        if not self._is_freewheeling:
            # Calculate the frame difference
            def tc_to_frames(t):
                return (t.hours * 3600 + t.minutes * 60 + t.seconds) * fps + t.frames

            base_frames = tc_to_frames(self._freewheel_base_timecode)
            new_frames = tc_to_frames(tc)

            frame_diff = new_frames - base_frames
            time_diff = current_time - self._freewheel_base_time

            # Expected frames based on time elapsed and direction
            expected_direction = -1 if not self._freewheel_base_timecode.count_up else 1
            expected_frames = time_diff * fps * expected_direction

            # Calculate how far off the incoming timecode is from expected
            deviation = abs(frame_diff - expected_frames)

            # Only reject jumps that are significantly off from expected
            # Allow up to 2 frames of deviation for normal timing variations
            max_deviation = 2

            # Also reject if direction changed without a legitimate reason
            direction_changed = (tc.count_up != self._freewheel_base_timecode.count_up)

            if deviation > max_deviation or direction_changed:
                # Jump is suspicious - enter freewheel mode
                self._is_freewheeling = True
                if self.debug:
                    _logger.debug(f"Freewheel: Suspicious jump detected (deviation: {deviation:.1f} frames, direction_change: {direction_changed})")

                # Fall through to generate freewheeled timecode below

        # If we're freewheeling (either just started or already in progress)
        if self._is_freewheeling:
            # Check if incoming timecode matches expected freewheled value
            time_since_base = current_time - self._freewheel_base_time
            expected_direction = -1 if not self._freewheel_base_timecode.count_up else 1
            expected_frame_offset = time_since_base * fps * expected_direction

            # Check if this incoming frame "catches up" to where we expect it to be
            def tc_to_frames(t):
                return (t.hours * 3600 + t.minutes * 60 + t.seconds) * fps + t.frames

            base_frames = tc_to_frames(self._freewheel_base_timecode)
            new_frames = tc_to_frames(tc)
            actual_frame_offset = new_frames - base_frames

            # If incoming matches expected within 2 frames, it caught up - resume normal
            if abs(actual_frame_offset - expected_frame_offset) <= 2:
                # Incoming timecode caught up to our freewheeled value
                self._is_freewheeling = False
                if self.debug:
                    _logger.debug(f"Freewheel: Incoming timecode caught up, resuming normal decoding")
                return True, None

            # Check if freewheel timeout expired - accept the jump
            if time_since_base > self._freewheel_timeout:
                self._is_freewheeling = False
                if self.debug:
                    _logger.debug(f"Freewheel: Timeout expired, accepting jump to new timecode")
                return True, None

            # Generate freewheeled timecode: increment from base at expected rate
            freewheeled = self._increment_timecode(self._freewheel_base_timecode, time_since_base, fps)
            return False, freewheeled

        # Not freewheeling and jump was acceptable
        return True, None

    def _increment_timecode(self, base_tc: Timecode, elapsed_seconds: float, fps: float) -> Timecode:
        """
        Increment a timecode by a time duration at a given frame rate.

        Args:
            base_tc: Base timecode to increment from
            elapsed_seconds: Time elapsed in seconds
            fps: Frame rate for calculation

        Returns:
            A new Timecode object incremented by the elapsed time
        """
        # Calculate total frame offset
        direction = -1 if not base_tc.count_up else 1
        total_frames = int(round(elapsed_seconds * fps * direction))

        # Convert base to total frames
        base_total = (base_tc.hours * 3600 + base_tc.minutes * 60 + base_tc.seconds) * fps + base_tc.frames
        new_total = base_total + total_frames

        # Handle negative (countdown past zero)
        if new_total < 0:
            new_total = abs(new_total)
            # For countdown that went past zero, flip direction
            count_up = True
        else:
            count_up = base_tc.count_up

        # Convert back to HH:MM:SS:FF
        new_frames = int(new_total % fps)
        remaining = new_total // fps
        new_seconds = int(remaining % 60)
        remaining = remaining // 60
        new_minutes = int(remaining % 60)
        new_hours = int(remaining // 60)

        # Create new timecode with same frame_rate and drop_frame setting as base
        return Timecode(
            hours=new_hours,
            minutes=new_minutes,
            seconds=new_seconds,
            frames=new_frames,
            frame_rate=base_tc.frame_rate,
            drop_frame=base_tc.drop_frame,
            count_up=count_up,
            user_bits=base_tc.user_bits
        )

    def _process_frames(self, frames: List[List[int]]) -> tuple[bool, int, int]:
        """
        Process decoded frames and extract timecode.

        Args:
            frames: List of 80-bit frame representations

        Returns:
            Tuple of (timecode_progressed: bool, valid_timecode_count: int, frames_rejected_for_hour: int)
        """
        timecode_progressed = False
        valid_timecode_count = 0
        frames_rejected_for_hour = 0  # Track frames rejected due to hour check

        for frame_bits in frames:
            try:
                timecode = Timecode.decode_80bit(frame_bits)
            except Exception as e:
                # Skip invalid frames - could be bit errors in the stream
                if self.debug:
                    _logger.debug(f"Failed to decode frame: {e}")
                continue

            if timecode:
                # Check if the hour is plausible given our history
                if not self._is_hour_plausible(timecode.hours):
                    # Skip this frame - likely a bit error
                    frames_rejected_for_hour += 1
                    if self.debug:
                        _logger.debug(f"Skipping frame with implausible hour: {timecode.hours}")
                    continue

                valid_timecode_count += 1

                # Freewheeling: check if this jump is reasonable
                # Returns (should_accept, freewheeled_timecode_or_None)
                current_time = time.time()
                should_accept, freewheeled_tc = self._is_timecode_jump_reasonable(timecode, current_time)

                # Determine which timecode to use
                if should_accept:
                    tc_to_use = timecode
                    # Update freewheel base to this new accepted timecode
                    self._freewheel_base_timecode = timecode
                    self._freewheel_base_time = current_time
                else:
                    tc_to_use = freewheeled_tc
                    # Don't update freewheel base - keep it for next iteration

                # Check if timecode is stuck (repeating same value)
                tc_value = (tc_to_use.hours, tc_to_use.minutes, tc_to_use.seconds, tc_to_use.frames)
                if self._last_timecode_value == tc_value:
                    self._stuck_frame_count += 1
                else:
                    self._stuck_frame_count = 0
                    self._last_timecode_value = tc_value
                    timecode_progressed = True

                # Update current timecode (the raw incoming value)
                self.current_timecode = timecode
                self.packets_received += 1
                self.last_update_time = current_time

                # Update hour confidence tracking based on raw incoming timecode
                # (we want to track the actual signal, not our freewheeled values)
                if self._last_valid_hour is not None:
                    if self._last_valid_hour // 10 == timecode.hours // 10:
                        # Same tens range, increase confidence
                        self._hour_confidence += 1
                    else:
                        # Hour tens changed, reset confidence
                        self._hour_confidence = 1
                else:
                    self._hour_confidence = 1
                self._last_valid_hour = timecode.hours

                # Send via OSC if enabled - send the timecode we're displaying
                # (which might be freewheeled)
                if self.osc_broadcaster and self._hour_confidence >= 3:
                    self.osc_broadcaster.send_timecode(tc_to_use)

                if self.callback:
                    self.callback(tc_to_use)

        return timecode_progressed, valid_timecode_count, frames_rejected_for_hour

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        try:
            self._audio_callback_impl(indata, frames, time_info, status)
        except Exception as e:
            # Log error but don't crash - audio callback must not raise exceptions
            _logger.error(f"Error in audio callback: {e}")
            # Reset state to recover
            self._in_detection_mode = True
            self.biphase = None
            self.current_timecode = None

    def _audio_callback_impl(self, indata: np.ndarray, frames, time_info, status):
        """Implementation of audio callback - wrapped in try/except by _audio_callback."""
        if status:
            _logger.warning(f"Audio status: {status}")

        # Extract the specified channel
        if indata.shape[1] > self.channel:
            samples = indata[:, self.channel].astype(np.float32)
        else:
            # Channel not available, use first channel
            samples = indata[:, 0].astype(np.float32)

        current_time = time.time()
        current_signal_max = np.max(np.abs(samples))

        if self.debug:
            _logger.debug(f"Callback: frames={len(samples)}, signal_max={current_signal_max:.4f}, "
                          f"in_detection={self._in_detection_mode}, biphase={'None' if self.biphase is None else 'exists'}")

        # SIMPLIFIED STATE MACHINE:
        # - Detection mode: accumulating samples to find frame rate
        # - Locked mode: have frame rate, processing timecodes
        # On ANY anomaly (signal loss, discontinuity), go back to detection mode

        # Check for signal loss - 500ms timeout (less aggressive for stability)
        if (not self._in_detection_mode and self.biphase is not None and
            self._last_valid_sample_time is not None and
            current_time - self._last_valid_sample_time > 0.5):
            # Signal lost - immediately enter detection mode
            self._in_detection_mode = True
            self.biphase = None
            self.frame_rate = None
            self.current_timecode = None
            self._last_valid_sample_time = None
            self._stuck_frame_count = 0
            self._last_timecode_value = None
            self._last_valid_hour = None  # Reset hour tracking to allow new hour values
            self._hour_confidence = 0
            self._detection_buffer.clear()
            self._detection_buffer_samples = 0
            # Reset freewheeling state
            self._freewheel_base_timecode = None
            self._freewheel_base_time = None
            self._is_freewheeling = False
            # Don't log warning every time - too noisy
            if self.debug:
                _logger.warning("[SIGNAL LOSS] Entering detection mode")
            # Fall through to detection mode

        # If in detection mode, accumulate samples for frame rate detection
        if self._in_detection_mode or self.biphase is None:

            # Track signal state transitions
            current_is_strong = current_signal_max >= 0.01
            signal_returned = current_is_strong and not self._last_signal_was_strong

            if signal_returned:
                # Signal just returned - purge any stale silent data from buffer
                self._detection_buffer.clear()
                self._detection_buffer_samples = 0
                _logger.info(f"[SIGNAL RETURN] Signal detected after silence, clearing stale buffer")

            self._last_signal_was_strong = current_is_strong

            # Check signal level of current block before adding to buffer
            if not current_is_strong:
                # Current block is essentially silent, skip it and reset
                self._detection_buffer.clear()
                self._detection_buffer_samples = 0
                return

            self._detection_buffer.append(samples)
            self._detection_buffer_samples += len(samples)

            # Limit buffer size to prevent runaway accumulation
            max_buffer_size = self._detection_threshold * 20  # Max 2 seconds
            if self._detection_buffer_samples > max_buffer_size:
                # Keep only the most recent samples
                self._detection_buffer = [self._detection_buffer[-1]]
                self._detection_buffer_samples = len(self._detection_buffer[-1])
                _logger.warning(f"Detection buffer overflow, truncating to {self._detection_buffer_samples} samples")

            # Check if we have enough samples for detection
            # Use a lower threshold for initial detection attempt, then increase
            min_samples = self._detection_threshold // 2  # Start with 0.05 seconds

            if self._detection_buffer_samples >= min_samples:
                # Combine buffer and detect frame rate
                combined = np.concatenate(self._detection_buffer)

                # Check signal level of combined buffer
                signal_max = np.max(np.abs(combined))
                if signal_max < 0.01:
                    # Signal too weak, clear buffer and keep accumulating fresh samples
                    self._detection_buffer.clear()
                    self._detection_buffer_samples = 0
                    if self.debug:
                        _logger.debug(f"Detection: combined signal too weak ({signal_max:.4f}), clearing buffer")
                    return

                # Try with progressively larger buffers until we find valid timecodes
                # or hit the maximum buffer size
                if self._detection_buffer_samples >= self._detection_threshold * 2:
                    # We've accumulated 2x the threshold without success - try harder
                    # Skip the first portion to try different alignments
                    skip_amount = len(combined) // 4
                    for offset in [0, skip_amount, skip_amount * 2, skip_amount * 3]:
                        test_buffer = combined[offset:] if offset > 0 else combined

                        detected_rate = detect_frame_rate(test_buffer, self.sample_rate)
                        self.frame_rate = detected_rate
                        self.biphase = BiphaseMDecoder(self.sample_rate, detected_rate)

                        # Process buffered samples
                        decoded_frames = self.biphase.process(test_buffer)

                        # Check if we got any frames with valid sync (even if timecode is invalid)
                        # This is more lenient than requiring valid timecodes
                        valid_frame_found = False
                        for frame_bits in decoded_frames:
                            if len(frame_bits) >= 80:
                                # Check for valid sync pattern (bits 64-79 must start with 0011)
                                sync_bits = frame_bits[64:68]
                                if sync_bits == [0, 0, 1, 1]:
                                    valid_frame_found = True
                                    if self.debug:
                                        _logger.debug(f"Detection: found valid sync at offset {offset}")
                                    break

                        if valid_frame_found:
                            # Try to get at least one valid timecode
                            self._process_frames(decoded_frames)
                            # If we got any valid timecode, exit detection mode
                            if self.current_timecode is not None:
                                self._in_detection_mode = False
                                self._last_valid_sample_time = current_time
                                self._stuck_frame_count = 0
                                self._last_timecode_value = None
                                # Initialize freewheeling state with first valid timecode
                                self._freewheel_base_timecode = self.current_timecode
                                self._freewheel_base_time = current_time
                                self._is_freewheeling = False
                                _logger.info(f"[DETECTION] Successfully locked on signal! (offset: {offset})")
                                # Clear buffer after success
                                self._detection_buffer.clear()
                                self._detection_buffer_samples = 0
                                return

                    # None of the offsets worked - clear some buffer and try again
                    self._detection_buffer.clear()
                    self._detection_buffer_samples = 0
                    self.biphase = None
                    self.frame_rate = None
                    _logger.warning("[DETECTION] No valid frames found at any offset, retrying...")
                    return

                # Normal detection path - try with current buffer
                detected_rate = detect_frame_rate(combined, self.sample_rate)
                self.frame_rate = detected_rate
                self.biphase = BiphaseMDecoder(self.sample_rate, detected_rate)
                _logger.info(f"[DETECTION] Detected frame rate: {detected_rate} fps")

                # Process buffered samples
                decoded_frames = self.biphase.process(combined)

                # Try to get at least one valid timecode (more lenient - just need valid sync)
                valid_frame_found = False
                for frame_bits in decoded_frames:
                    if len(frame_bits) >= 80:
                        sync_bits = frame_bits[64:68]
                        if sync_bits == [0, 0, 1, 1]:
                            valid_frame_found = True
                            if self.debug:
                                _logger.debug(f"Detection: found valid sync")
                            break

                if valid_frame_found:
                    # Try to process frames - may get valid timecodes
                    self._process_frames(decoded_frames)
                    # If we got any valid timecode, exit detection mode
                if self.current_timecode is not None:
                    self._in_detection_mode = False
                    self._last_valid_sample_time = current_time
                    self._stuck_frame_count = 0
                    self._last_timecode_value = None
                    # Initialize freewheeling state with first valid timecode
                    self._freewheel_base_timecode = self.current_timecode
                    self._freewheel_base_time = current_time
                    self._is_freewheeling = False
                    _logger.info("[DETECTION] Successfully locked on signal!")
                    # Clear buffer after success
                    self._detection_buffer.clear()
                    self._detection_buffer_samples = 0
                else:
                    # No valid timecodes yet - keep accumulating, don't clear buffer
                    # Stay in detection mode but keep the decoder
                    if self.debug:
                        _logger.debug(f"Detection: no valid timecodes yet, buffer size: {self._detection_buffer_samples}")
                return
            else:
                return  # Still accumulating

        # Decode to frames - but skip if signal is too weak
        # This prevents the decoder buffer from getting corrupted with silence
        current_signal_max = np.max(np.abs(samples))

        if current_signal_max < 0.01:
            # Signal too weak - don't process these samples
            return

        decoded_frames = self.biphase.process(samples)

        # Process frames
        if decoded_frames:
            timecode_progressed, valid_count, rejected_for_hour = self._process_frames(decoded_frames)

            # If we're getting frames (even if some are rejected), consider
            # the signal valid - update _last_valid_sample_time to avoid false signal loss
            if valid_count > 0 or rejected_for_hour > 0:
                self._last_valid_sample_time = current_time

            # Check for stuck frames - same timecode repeating without progress
            # If we see the same frame 5+ times with strong signal, decoder is corrupted
            if (not self._in_detection_mode and
                self._stuck_frame_count >= 5 and
                current_signal_max >= 0.05):
                # Reset biphase decoder to clear corrupted state
                self.biphase.reset()
                self._stuck_frame_count = 0
                self._last_timecode_value = None
                _logger.warning("[STUCK FRAMES] Resetting biphase decoder to recover")
                return

            # Check for frames but no valid timecodes - decoder is misaligned
            if valid_count == 0 and not self._in_detection_mode and current_signal_max >= 0.05:
                # Track consecutive invalid frame decodes
                if not hasattr(self, '_invalid_frame_count'):
                    self._invalid_frame_count = 0
                self._invalid_frame_count += 1

                # After MORE consecutive callbacks with frames but no valid timecodes, reset
                # Increased from 3 to 10 to be more tolerant at high hours where bit errors
                # in the hour field are more common
                if self._invalid_frame_count >= 10:
                    self.biphase.reset()
                    self._stuck_frame_count = 0
                    self._last_timecode_value = None
                    self._invalid_frame_count = 0
                    _logger.warning("[INVALID FRAMES] Frames produced but no valid timecodes, resetting biphase decoder")
                return

            # Reset invalid frame counter on valid decode
            if valid_count > 0:
                if hasattr(self, '_invalid_frame_count'):
                    self._invalid_frame_count = 0
        else:
            # No frames decoded even though signal is strong
            # This could mean the biphase decoder is corrupted or misaligned
            # Instead of entering detection mode (which loses frame rate context),
            # just reset the biphase decoder and let it re-sync
            if not self._in_detection_mode and current_signal_max >= 0.05:
                # Track consecutive empty decodes
                if not hasattr(self, '_empty_decode_count'):
                    self._empty_decode_count = 0
                self._empty_decode_count += 1

                # After 3 consecutive empty decodes with strong signal, reset the decoder
                if self._empty_decode_count >= 3:
                    self.biphase.reset()
                    self._stuck_frame_count = 0
                    self._last_timecode_value = None
                    self._empty_decode_count = 0
                    _logger.warning("[NO FRAMES] Resetting biphase decoder to recover")
                return

            # Reset empty decode counter on successful decode
            if hasattr(self, '_empty_decode_count'):
                self._empty_decode_count = 0
        # Note: We DON'T update _last_valid_sample_time on empty decodes
        # This means 500ms with no valid frames = signal loss = enter detection mode

    def start(self):
        """Start decoding from audio input."""
        if self.stream is not None:
            return  # Already running

        # Request all available channels so user can select which one to use
        # Use fixed blocksize for consistent timing (especially important on Windows)
        # 0 = let system decide, but this causes jitter on Windows audio drivers
        self.stream = sd.InputStream(
            device=self.device,
            channels=2,  # Request stereo to get both channels
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=2048,  # Fixed buffer size for consistent timing
            latency='low',   # Request low latency mode
        )

        self.stream.start()

    def stop(self):
        """Stop decoding."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Stop OSC re-distribution
        if self.osc_broadcaster:
            self.osc_broadcaster.disable()

    def get_timecode(self, send_osc: bool = False) -> Optional[Timecode]:
        """
        Get the current timecode.

        Args:
            send_osc: If True, also send OSC broadcast with this timecode

        Returns:
            Timecode object, or None if no valid data received recently
        """
        current_time = time.time()

        # Consider data stale if no updates for 1 second
        if self.last_update_time is None:
            return None

        if current_time - self.last_update_time > 1.0:
            return None

        # If we have a freewheel base, calculate and return the freewheeled timecode
        tc = None
        if self._freewheel_base_timecode is not None:
            fps = self.frame_rate if self.frame_rate else (self._freewheel_base_timecode.fps if self._freewheel_base_timecode.fps else 30.0)
            elapsed = current_time - self._freewheel_base_time
            tc = self._increment_timecode(self._freewheel_base_timecode, elapsed, fps)
        else:
            # Fall back to current_timecode if no freewheel base
            tc = self.current_timecode

        # Send OSC if requested and we have sufficient confidence
        if send_osc and tc and self.osc_broadcaster and self._hour_confidence >= 3:
            self.osc_broadcaster.send_timecode(tc)

        return tc

    def get_statistics(self) -> dict:
        """
        Get decoder statistics.

        Returns:
            Dict with current status
        """
        return {
            "packets_received": self.packets_received,
            "timecode": self.get_timecode(),
        }


def format_timecode(tc: Timecode) -> str:
    """Format timecode for display (with ; separator for drop-frame)."""
    if tc is None:
        return "--:--:--:--"

    direction = "▲" if tc.count_up else "▼"
    separator = ";" if tc.is_drop_frame else ":"
    return f"{direction} {tc.hours:02d}:{tc.minutes:02d}:{tc.seconds:02d}{separator}{tc.frames:02d}"


def detect_frame_rate(
    samples: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Auto-detect frame rate from audio samples by trying all rates.

    The detection works by:
    1. Finding edges (zero-crossings) in the audio signal
    2. Measuring the actual period between edges
    3. Comparing to expected periods for each frame rate
    4. Also checking bits 10-11 for drop-frame and color frame flags

    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Detected frame rate, or 30.0 as default
    """
    # First, analyze edge timing to narrow down candidates
    # Find edges (zero-crossings)
    edges = []
    for i in range(1, len(samples)):
        if (samples[i-1] >= 0 and samples[i] < 0) or (samples[i-1] < 0 and samples[i] >= 0):
            edges.append(i)

    if len(edges) < 20:
        return 30.0  # Not enough data

    # Calculate periods between consecutive edges
    periods = []
    for i in range(1, min(len(edges), 100)):
        periods.append(edges[i] - edges[i-1])

    if not periods:
        return 30.0

    # Expected samples per half-bit period for each frame rate
    # At 30fps: 48000 / (80 * 30) / 2 = 10 samples per half-bit
    # At 24fps: 48000 / (80 * 24) / 2 = 12.5 samples per half-bit
    # At 25fps: 48000 / (80 * 25) / 2 = 12 samples per half-bit
    # At 29.97fps: 48000 / (80 * 29.97) / 2 ≈ 10.01 samples per half-bit
    # At 23.98fps: 48000 / (80 * 23.98) / 2 ≈ 12.51 samples per half-bit

    expected_half_bits = {
        30.0: sample_rate / (80 * 30) / 2,
        29.97: sample_rate / (80 * 29.97) / 2,
        25.0: sample_rate / (80 * 25) / 2,
        24.0: sample_rate / (80 * 24) / 2,
        23.98: sample_rate / (80 * 23.98) / 2,
    }

    # Score each frame rate by counting how many measured periods fall within tolerance
    # This is more robust than just taking the most common period
    timing_scores = {}
    for fps, expected_period in expected_half_bits.items():
        # Count periods within ±1.5 samples of expected (tolerant to timing variations)
        tolerance = 1.5
        matching_count = sum(1 for p in periods if abs(p - expected_period) <= tolerance)
        # Also give partial credit for periods within ±3 samples
        partial_count = sum(1 for p in periods if 1.5 < abs(p - expected_period) <= 3)
        timing_scores[fps] = matching_count * 2 + partial_count

    # Sort by score (higher is better)
    timing_candidates = sorted(timing_scores.items(), key=lambda x: x[1], reverse=True)

    # Now verify the top candidates by actually decoding and checking bits 10-11
    # Only check the top 2 timing candidates to avoid unnecessary work
    frame_rate_configs = [
        (30.0, False, False),
        (25.0, False, True),
        (29.97, True, False),
        (24.0, False, False),
        (23.98, False, False),
        (29.97, False, False),
        (30.0, True, False),
    ]

    best_rate = 30.0
    best_score = -1

    # Prioritize frame rates that match the measured timing
    # Give top 2 timing candidates priority
    priority_fps = [fps for fps, _ in timing_candidates[:2]]

    # Try both normal and inverted polarity
    for inverted in (False, True):
        test_samples = -samples if inverted else samples

        for test_rate, bit10_drop, bit11_color in frame_rate_configs:
            decoder = BiphaseMDecoder(sample_rate, test_rate)
            frames = decoder.process(test_samples.copy())

            if not frames:
                continue

            # Analyze the decoded frames
            matching_bits = 0
            valid_timecodes = 0

            for i in range(min(len(frames), 50)):
                frame_bits = frames[i]
                if len(frame_bits) < 80:
                    continue

                # Check if bits 10 and 11 match what we expect for this rate
                actual_bit10 = frame_bits[10]
                actual_bit11 = frame_bits[11]

                if actual_bit10 == (1 if bit10_drop else 0) and actual_bit11 == (1 if bit11_color else 0):
                    matching_bits += 1

                # Also count valid timecodes
                tc = Timecode.decode_80bit(frame_bits)
                if tc:
                    valid_timecodes += 1

            # Score: prioritize matching bits, then valid timecode count
            # Add timing bonus if this rate matches our measured timing
            timing_bonus = 50 if test_rate in priority_fps else 0
            polarity_penalty = 0 if not inverted else 10
            # Prefer 30fps as tiebreaker for ambiguous cases (most common, and 24/30 encode identically)
            fps_preference = 5 if test_rate == 30.0 else 0
            score = matching_bits * 100 + valid_timecodes + timing_bonus + fps_preference - polarity_penalty

            if score > best_score:
                best_score = score
                best_rate = test_rate

    return best_rate


def decode_file(
    file_path: str,
    sample_rate: Optional[int] = None,
    frame_rate: Optional[float] = None,
) -> tuple:
    """
    Decode SMPTE/LTC from an audio file.

    Reads the beginning and end of the file to determine start/end timecodes
    and actual direction (by comparing timecodes).

    Args:
        file_path: Path to audio file
        sample_rate: Expected sample rate (None to use file's native rate)
        frame_rate: Frame rate (None for auto-detect)

    Returns:
        Tuple of (first_timecode, last_timecode, detected_frame_rate, file_duration_seconds)
    """
    # Read the entire file to get total length (soundfile is fast for this)
    all_samples, sr = sf.read(file_path)
    total_samples = len(all_samples)
    total_duration = total_samples / sr

    # Use first channel if stereo
    if len(all_samples.shape) > 1:
        all_samples = all_samples[:, 0]

    # If sample_rate is specified and differs from file rate, resample
    if sample_rate is not None and sr != sample_rate:
        from scipy import signal
        all_samples = signal.resample(all_samples, int(len(all_samples) * sample_rate / sr))
        sr = sample_rate

    # Auto-detect frame rate and polarity from the beginning if not specified
    start_samples = all_samples[:int(sr * 2)]
    if frame_rate is None:
        frame_rate = detect_frame_rate(start_samples, sr)

    # Detect polarity by trying both and seeing which produces MORE valid timecodes
    use_inverted = False
    test_decoder = BiphaseMDecoder(sr, frame_rate)
    test_frames_normal = test_decoder.process(start_samples.copy())
    valid_count_normal = sum(1 for fb in test_frames_normal if Timecode.decode_80bit(fb))

    test_decoder.reset()
    test_frames_inverted = test_decoder.process(-start_samples.copy())
    valid_count_inverted = sum(1 for fb in test_frames_inverted if Timecode.decode_80bit(fb))

    # Use the polarity that produces more valid timecodes
    if valid_count_inverted > valid_count_normal:
        use_inverted = True

    # Apply detected polarity if needed
    if use_inverted:
        all_samples = -all_samples

    # Process the file in chunks to maintain biphase decoder sync
    decoder = BiphaseMDecoder(sr, frame_rate)
    chunk_size = int(sr * 1)  # 1 second chunks

    all_timecodes = []
    for i in range(0, len(all_samples), chunk_size):
        chunk = all_samples[i:i+chunk_size]
        frames = decoder.process(chunk)
        for fb in frames:
            tc = Timecode.decode_80bit(fb)
            if tc:
                all_timecodes.append(tc)

    first_tc = all_timecodes[0] if all_timecodes else None
    last_tc = all_timecodes[-1] if all_timecodes else None

    return first_tc, last_tc, frame_rate, total_duration


def main():
    parser = argparse.ArgumentParser(
        description="Decode SMPTE/LTC timecode from audio input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i test.wav                  # Decode from file
  %(prog)s                              # Decode from live audio
  %(prog)s -d 2                         # Use device 2
  %(prog)s --list-devices               # Show audio devices

The decoder automatically detects:
- Sample rate (from device default)
- Frame rate (from timing analysis, or use -r to specify)
- Count-up mode (standard SMPTE, shows ▲)
- Countdown mode (countdown, shows ▼)

Direction is indicated by bit 60:
- Bit 60 = 1: Counting up (standard SMPTE)
- Bit 60 = 0: Counting down (countdown mode)
        """,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Decode from file instead of live audio",
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        help="Audio input device number (default: system default)",
    )
    parser.add_argument(
        "-c", "--channel",
        type=int,
        default=0,
        help="Audio channel to listen to (0=left, 1=right, default: 0)",
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=None,
        help="Sample rate in Hz (default: auto-detect from device)",
    )
    parser.add_argument(
        "-r", "--frame-rate",
        type=float,
        default=None,
        choices=[23.98, 24.0, 25.0, 29.97, 30.0],
        help="Frame rate (default: auto-detect)",
    )
    parser.add_argument(
        "-l", "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with detailed logging",
    )
    parser.add_argument(
        "--osc",
        action="store_true",
        help="Enable OSC broadcasting of timecode",
    )
    parser.add_argument(
        "--osc-address",
        type=str,
        default="255.255.255.255",
        help="OSC broadcast address (default: 255.255.255.255)",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=9988,
        help="OSC UDP port (default: 9988)",
    )

    args = parser.parse_args()

    if args.list_devices:
        print("Audio Input Devices:")
        print("-" * 60)
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']}")
        return

    # File decoding mode
    if args.input:
        print(f"Decoding from file: {args.input}")
        print("-" * 40)

        first_tc, last_tc, detected_rate, duration = decode_file(
            args.input, args.sample_rate, args.frame_rate
        )

        if first_tc is None:
            print("No SMPTE/LTC frames detected.", file=sys.stderr)
            sys.exit(1)

        # Show detected frame rate
        if args.frame_rate is None:
            print(f"Auto-detected frame rate: {detected_rate} fps")
            print()

        # Determine actual direction by comparing first and last timecodes
        # (ignoring bit 60, since that may not be set correctly in standard LTC)
        fps = first_tc.fps

        def tc_to_frames(tc):
            total = (tc.hours * 3600 + tc.minutes * 60 + tc.seconds) * fps + tc.frames
            return int(total)

        first_frames = tc_to_frames(first_tc)
        if last_tc is not None:
            last_frames = tc_to_frames(last_tc)
            # Determine direction from actual timecode values
            counting_up = last_frames >= first_frames
        else:
            # No end timecode found, fall back to bit 60
            counting_up = first_tc.count_up
            last_frames = first_frames

        direction_text = "Counting up" if counting_up else "Counting down"
        direction_symbol = "▲" if counting_up else "▼"

        # Calculate duration from timecodes
        if counting_up:
            duration_frames = last_frames - first_frames
        else:
            duration_frames = first_frames - last_frames

        # Convert duration frames to HH:MM:SS:FF
        dur_hours = int(duration_frames // (fps * 3600))
        duration_frames %= fps * 3600
        dur_minutes = int(duration_frames // (fps * 60))
        duration_frames %= fps * 60
        dur_seconds = int(duration_frames // fps)
        dur_ff = int(duration_frames % fps)

        # Use ; separator for drop-frame timecodes
        separator = ";" if first_tc.is_drop_frame else ":"
        tc_duration = f"{dur_hours:02d}:{dur_minutes:02d}:{dur_seconds:02d}{separator}{dur_ff:02d}"

        # Display summary
        print(f"Start timecode: {format_timecode(first_tc)}")
        if last_tc is not None:
            print(f"End timecode:   {direction_symbol} {last_tc.hours:02d}:{last_tc.minutes:02d}:{last_tc.seconds:02d}{separator}{last_tc.frames:02d}")
        else:
            print("End timecode:   (not found in file)")
        print(f"Duration:       {tc_duration}")
        print(f"Direction:      {direction_text}")
        print(f"Frame rate:     {detected_rate} fps")
        return

    # Live decoding mode
    print("Decoding SMPTE/LTC from live audio input...")
    if args.device is not None:
        print(f"Using device {args.device}")
    print(f"Using channel {args.channel} ({'left' if args.channel == 0 else 'right'})")
    if args.frame_rate is None:
        print("Frame rate: auto-detecting...")
    else:
        print(f"Frame rate: {args.frame_rate} fps")
    print("Press Ctrl+C to stop.")
    print("-" * 40)

    # Initialize OSC broadcaster if enabled
    osc_broadcaster = None
    if args.osc:
        osc_broadcaster = OSCBroadcaster(address=args.osc_address, port=args.osc_port)
        osc_broadcaster.enable()
        print(f"OSC broadcasting: {args.osc_address}:{args.osc_port}")
        print(f"  Paths: /flextc/ltc, /flextc/count")
        print("-" * 40)

    decoder = Decoder(
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        device=args.device,
        channel=args.channel,
        debug=args.verbose,
    )

    def display_callback(tc: Timecode):
        pass  # We'll display in the main loop

    # Track detection state for display
    last_was_detecting = False
    detection_shown = args.frame_rate is not None

    try:
        decoder.start()

        last_display = ""
        while True:
            time.sleep(0.1)

            tc = decoder.get_timecode()
            stats = decoder.get_statistics()

            # Send via OSC if enabled
            if tc and osc_broadcaster:
                osc_broadcaster.send_timecode(tc)

            # Check if we're in detection mode
            is_detecting = decoder._in_detection_mode or decoder.biphase is None

            # Show detection complete message when we first lock on
            if not is_detecting and last_was_detecting:
                print(f"\nAuto-detected frame rate: {decoder.frame_rate} fps")
                print("-" * 40)
                detection_shown = True

            if tc:
                display = f"\r{format_timecode(tc)}  (frames: {stats['packets_received']})  "
                print(display, end="", flush=True)
                last_display = display
            else:
                if is_detecting:
                    # Still detecting or re-detecting after signal loss
                    waiting_msg = f"\rDetecting frame rate... ({int(decoder._detection_buffer_samples / decoder.sample_rate * 10) / 10}s / 0.1s)  "
                    print(waiting_msg, end="", flush=True)
                    last_was_detecting = True
                elif time.time() - (decoder.last_update_time or 0) > 0.5:
                    print(f"\r{' ' * 60}", end="", flush=True)
                    print(f"\r--:--:--:--  (waiting for signal...)  ", end="", flush=True)

            last_was_detecting = is_detecting

    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        decoder.stop()
        if osc_broadcaster:
            osc_broadcaster.disable()


if __name__ == "__main__":
    main()
