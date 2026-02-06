#!/usr/bin/env python3
"""
RLTC Encoder CLI - Generate countdown timecode audio files.
"""

import sys
from pathlib import Path

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rltc import RLTCEncoder, parse_duration


@click.command()
@click.argument(
    "duration",
    type=str,
    help="Duration (e.g., '5m', '30s', '1:30', '5000ms')",
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default="rltc_countdown.wav",
    help="Output WAV file path",
)
@click.option(
    "-r", "--rate",
    type=float,
    default=30.0,
    help="Packet rate in Hz (default: 30)",
)
@click.option(
    "-a", "--amplitude",
    type=float,
    default=0.7,
    help="Amplitude 0.0-1.0 (default: 0.7)",
)
@click.option(
    "-s", "--sample-rate",
    type=int,
    default=44100,
    help="Sample rate in Hz (default: 44100)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output",
)
def main(duration: str, output: str, rate: float, amplitude: float, sample_rate: int, verbose: bool):
    """
    Generate an RLTC countdown audio file.

    Examples:

        rltc_encode 5m -o countdown.wav

        rltc_encode 1:30 -r 25 -o 90sec.wav

        rltc_encode 30s --rate 50 --output test.wav
    """
    # Parse duration
    try:
        duration_ms = parse_duration(duration)
    except Exception as e:
        click.echo(f"Error parsing duration: {e}", err=True)
        sys.exit(1)

    if verbose:
        duration_sec = duration_ms / 1000
        click.echo(f"Generating {duration_sec:.1f}s countdown audio...")
        click.echo(f"  Output: {output}")
        click.echo(f"  Sample rate: {sample_rate} Hz")
        click.echo(f"  Packet rate: {rate} Hz")
        click.echo(f"  Amplitude: {amplitude}")

    # Create encoder and generate file
    encoder = RLTCEncoder(sample_rate=sample_rate)

    try:
        encoder.generate_to_file(
            output_path=output,
            duration_ms=duration_ms,
            packet_rate_hz=rate,
            amplitude=amplitude,
        )
        click.echo(f"✓ Generated {output}")
    except Exception as e:
        click.echo(f"Error generating file: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
