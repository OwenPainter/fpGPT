#!/usr/bin/env python3
"""Raw UART CLI for talking to the fpGPT FPGA via the ESP32-S2 USB bridge.

The DE1-SoC is reached through an ESP32-S2 native-USB CDC port
(default /dev/ttyACM0 at 115200 8N1).  The FPGA starts generating on a CR
(0x0D) or LF (0x0A) trigger and streams raw characters back
(GEN_TOKENS=16 per prompt in the current bring-up ROM).

Examples
--------
Send one prompt and print the raw reply::

    python3 host_gui/fpga_uart.py 'hello'

Send the same prompt 5 times::

    python3 host_gui/fpga_uart.py --count 5 'hello'

Exercise the ESP32 GPIO17<->GPIO18 jumper::

    python3 host_gui/fpga_uart.py --loopback

Use a different port / longer read window::

    python3 host_gui/fpga_uart.py -p /dev/ttyUSB1 -t 3.0 --hex 'hi'

Only the Python standard library is used (termios/select/fcntl).
"""

import argparse
import errno
import fcntl
import os
import select
import sys
import termios
import time

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 1.0
GEN_TOKENS = 16

# termios constants that may be missing on some platforms.
TIOCM_DTR = getattr(termios, "TIOCM_DTR", 0x002)
TIOCMGET = getattr(termios, "TIOCMGET", 0x5415)
TIOCMSET = getattr(termios, "TIOCMSET", 0x5418)


def baud_constant(baud):
    """Map a numeric baud rate to its termios constant."""
    name = "B{}".format(baud)
    if hasattr(termios, name):
        return getattr(termios, name)
    speeds = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
    }
    if baud in speeds:
        return speeds[baud]
    raise ValueError("unsupported baud rate: {}".format(baud))


def open_port(path, baud, assert_dtr=True):
    """Open *path* in raw 8N1 mode, returning the fd.

    DTR is asserted (without toggling it) so the ESP32-S2 native USB does
    not see a reset edge.  Raises OSError with a friendly message on
    failure.
    """
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except FileNotFoundError:
        raise OSError(
            "serial port {} not found. Is the ESP32-S2 USB bridge "
            "plugged in? (check `ls /dev/ttyACM*`)".format(path)
        )
    except PermissionError:
        raise OSError(
            "permission denied opening {}. Add your user to the "
            "'dialout' group or run with sufficient privileges.".format(path)
        )
    except OSError as exc:
        if exc.errno == errno.EBUSY:
            raise OSError(
                "serial port {} is busy (another program holds it); "
                "close it and retry.".format(path)
            )
        raise OSError("could not open {}: {}".format(path, exc))

    try:
        # Keep the fd non-blocking; select() provides the timing.
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)

        attrs = termios.tcgetattr(fd)
        iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs

        # Raw mode: no translation, no echo, no signals.
        iflag &= ~(
            termios.IGNBRK
            | termios.BRKINT
            | termios.PARMRK
            | termios.ISTRIP
            | termios.INLCR
            | termios.IGNCR
            | termios.ICRNL
            | termios.IXON
            | termios.IXOFF
            | termios.IXANY
        )
        oflag &= ~termios.OPOST
        lflag &= ~(
            termios.ECHO
            | termios.ECHONL
            | termios.ICANON
            | termios.ISIG
            | termios.IEXTEN
        )
        cflag &= ~(termios.CSIZE | termios.PARENB)
        cflag |= termios.CS8 | termios.CREAD | termios.CLOCAL
        cc[termios.VMIN] = 0
        cc[termios.VTIME] = 0

        speed = baud_constant(baud)
        attrs = [iflag, oflag, cflag, lflag, speed, speed, cc]
        termios.tcsetattr(fd, termios.TCSANOW, attrs)

        if assert_dtr:
            set_dtr(fd, True)

        termios.tcflush(fd, termios.TCIOFLUSH)
        return fd
    except Exception:
        os.close(fd)
        raise


def set_dtr(fd, state):
    """Assert/clear DTR via TIOCMSET without toggling other lines."""
    try:
        bits = fcntl.ioctl(fd, TIOCMGET, 0)
        if state:
            bits |= TIOCM_DTR
        else:
            bits &= ~TIOCM_DTR
        fcntl.ioctl(fd, TIOCMSET, bits)
    except OSError:
        # Some CDC drivers don't implement modem ioctls; harmless.
        pass


def drain(fd, timeout):
    """Read until no data arrives for *timeout* seconds; return bytes."""
    chunks = []
    deadline = time.monotonic() + timeout
    quiet_since = time.monotonic()
    while True:
        remaining = min(0.05, max(0.0, deadline - time.monotonic()))
        if deadline - time.monotonic() <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if ready:
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                data = b""
            if data:
                chunks.append(data)
                quiet_since = time.monotonic()
                # Extend the deadline while data keeps flowing.
                deadline = quiet_since + timeout
        else:
            # No data in this slice: stop once we've been quiet long enough.
            if time.monotonic() - quiet_since >= timeout:
                break
    return b"".join(chunks)


def send_and_read(fd, prompt, timeout, inter_prompt_delay=0.05):
    """Write prompt + CR and read the reply for up to *timeout* seconds."""
    os.write(fd, prompt.encode("utf-8", "replace") + b"\r")
    time.sleep(inter_prompt_delay)
    return drain(fd, timeout)


def loopback(fd, timeout, size=GEN_TOKENS):
    """Send a burst and count how many bytes echo back.

    Useful for verifying the ESP32 GPIO17<->GPIO18 jumper.
    """
    payload = bytes((0x41 + (i % 26)) for i in range(size))
    os.write(fd, payload)
    time.sleep(0.05)
    received = drain(fd, timeout)
    return payload, received


def format_reply(data, as_hex):
    if as_hex:
        return " ".join("{:02x}".format(b) for b in data)
    return repr(data.decode("latin-1"))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Send a prompt to the fpGPT FPGA over the ESP32-S2 USB-UART "
            "bridge and print the raw reply."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="text to send (CR is appended automatically)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_PORT,
        help="serial device (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help="baud rate (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds to wait for a reply (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="send the prompt this many times (default: %(default)s)",
    )
    parser.add_argument(
        "--no-dtr",
        action="store_true",
        help="do not assert DTR on open",
    )
    parser.add_argument(
        "--hex",
        action="store_true",
        help="print reply bytes as hex",
    )
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="send a %d-byte burst and report how many bytes return "
        "(ESP32 GPIO17<->GPIO18 jumper test)" % GEN_TOKENS,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress per-prompt reply printing (still prints counts)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.count < 1:
        print("error: --count must be >= 1", file=sys.stderr)
        return 2

    try:
        fd = open_port(args.port, args.baud, assert_dtr=not args.no_dtr)
    except (OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    try:
        if args.loopback:
            sent, received = loopback(fd, args.timeout)
            print("loopback: sent {} bytes, received {} bytes".format(
                len(sent), len(received)))
            print("sent:     {}".format(format_reply(sent, args.hex)))
            print("received: {}".format(format_reply(received, args.hex)))
            ok = len(received) >= len(sent)
            print("result:   {}".format("PASS" if ok else "FAIL"))
            return 0 if ok else 1

        total_read = 0
        for i in range(args.count):
            reply = send_and_read(fd, args.prompt, args.timeout)
            total_read += len(reply)
            if not args.quiet:
                prefix = "" if args.count == 1 else "[{}/{}] ".format(
                    i + 1, args.count)
                print("{}{}".format(prefix, format_reply(reply, args.hex)))
        if args.quiet or args.count > 1:
            print("read {} byte(s) over {} prompt(s)".format(
                total_read, args.count))
        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())