#!/usr/bin/env python3
"""
RLTC Decoder CLI - Real-time decode countdown from audio input.
"""

import sys
import time
from pathlib import Path

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rltc import RLTCDecoder, decode_file


def format_time(ms: int) -> str:
    """Format milliseconds as HH:MM:SS.mmm"""
    if ms is None:
        return "--:--:--.---"

    seconds = ms // 1000
    milliseconds = ms % 1000

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


@click.command()
@click.option(
    "-i", "--input",
    type=click.Path(exists=True),
    help="Decode from file instead of live audio",
)
@click.option(
    "-d", "--device",
    type=int,
    help="Audio input device number (default: system default)",
)
@click.option(
    "-s", "--sample-rate",
    type=int,
    default=44100,
    help="Sample rate in Hz (default: 44100)",
)
@click.option(
    "-l", "--list-devices",
    is_flag=True,
    help="List available audio input devices",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output with statistics",
)
def main(input: str | None, device: int | None, sample_rate: int, list_devices: bool, verbose: bool):
    """
    Decode RLTC countdown from audio input.

    Examples:

        rltc_decode                    # Decode from default input

        rltc_decode -d 2               # Use device 2

        rltc_decode -i test.wav        # Decode from file

        rltc_decode --list-devices     # Show audio devices
    """
    if list_devices:
        import sounddevice as sd
        click.echo("Audio Input Devices:")
        click.echo("-" * 60)
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                click.echo(f"  [{i}] {dev['name']}")
        return

    # File decoding mode
    if input:
        click.echo(f"Decoding from file: {input}")
        click.echo("-" * 40)

        results = decode_file(input, sample_rate)

        if not results:
            click.echo("No RLTC packets detected.", err=True)
            sys.exit(1)

        # Show results
        click.echo(f"Detected {len(results)} packets:")
        for timestamp, time_ms in results[:10]:  # Show first 10
            click.echo(f"  {timestamp:6.2f}s -> {format_time(time_ms)}")

        if len(results) > 10:
            click.echo(f"  ... and {len(results) - 10} more")

        # Show time range
        first_time = results[0][1]
        last_time = results[-1][1]
        click.echo(f"\nTime range: {format_time(first_time)} to {format_time(last_time)}")
        return

    # Live decoding mode
    click.echo("Decoding RLTC from live audio input...")
    if device is not None:
        click.echo(f"Using device {device}")
    click.echo("Press Ctrl+C to stop.")
    click.echo("-" * 40)

    last_display_time = 0
    packet_count = 0
    last_time_ms = None

    def time_callback(time_ms: int):
        nonlocal last_time_ms, packet_count
        last_time_ms = time_ms
        packet_count += 1

    decoder = RLTCDecoder(
        sample_rate=sample_rate,
        device=device,
        callback=time_callback,
    )

    try:
        decoder.start()

        while True:
            time.sleep(0.1)  # Update display 10x per second

            current_time = decoder.get_time_remaining()

            if current_time is not None:
                # Update display
                time_str = format_time(current_time)
                click.echo(f"\r⏱ {time_str}  (packets: {packet_count})", nl=False)
                last_display_time = time.time()
            else:
                # No signal
                if time.time() - last_display_time > 0.5:
                    click.echo(f"\r⏱ --:--:--.---  (waiting for signal...)  ", nl=False)
                    last_time_ms = None

            if verbose and packet_count > 0 and packet_count % 100 == 0:
                stats = decoder.get_statistics()
                click.echo(f"\nStats: {stats}")

    except KeyboardInterrupt:
        click.echo("\n\nStopped.")
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)
    finally:
        decoder.stop()


if __name__ == "__main__":
    main()
