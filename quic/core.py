"""Deterministic QUIC congestion control and loss recovery kernel.

The module models the sending half of a QUIC connection without touching a
real socket.  Time and acknowledgements are injected by the caller, one step
at a time, so a whole transfer can be replayed deterministically from a test.
It covers the RTT estimator with ACK delay compensation, the RTO and PTO
figures with exponential back off, slow start and congestion avoidance, packet
and time threshold loss detection, and the recovery period that guards the
congestion window against repeated reductions.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

MSS = 1200
INITIAL_WINDOW = 10 * MSS
MINIMUM_WINDOW = 2 * MSS
INITIAL_RTT = 0.333
TIMER_GRANULARITY = 0.001
MAX_ACK_DELAY = 0.025
MIN_RTO = 0.2
MAX_RTO = 60.0
ALPHA = 0.125
BETA = 0.25
K_PACKET_THRESHOLD = 3
TIME_THRESHOLD = 9.0 / 8.0
LOSS_REDUCTION_FACTOR = 0.5


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_time(now: object) -> float:
    if not _is_number(now) or now < 0:
        raise ValueError("time must be a finite non-negative number")
    return float(now)


class QuicError(Exception):
    """Base class for the errors raised by this module."""


class InvalidPacketError(QuicError, ValueError):
    """Raised when a packet number, size or acknowledgement is out of range."""


class SentPacket:
    """A packet that has been sent and not acknowledged yet."""

    __slots__ = ("packet_number", "sent_time", "size", "ack_eliciting")

    def __init__(self, packet_number: int, sent_time: float, size: int = MSS,
                 ack_eliciting: bool = True) -> None:
        self.packet_number = packet_number
        self.sent_time = sent_time
        self.size = size
        self.ack_eliciting = ack_eliciting

    @property
    def in_flight(self) -> bool:
        """Only ack eliciting packets count against the congestion window."""
        return self.ack_eliciting

    def __repr__(self) -> str:
        return "SentPacket(packet_number=%d, sent_time=%r, size=%d)" % (
            self.packet_number,
            self.sent_time,
            self.size,
        )


class RttEstimator:
    """Smoothed RTT, variance and minimum RTT, plus the RTO and PTO figures."""

    def __init__(self, initial_rtt: float = INITIAL_RTT, max_ack_delay: float = MAX_ACK_DELAY,
                 granularity: float = TIMER_GRANULARITY, min_rto: float = MIN_RTO,
                 max_rto: float = MAX_RTO) -> None:
        if not _is_number(initial_rtt) or initial_rtt <= 0:
            raise ValueError("initial_rtt must be a positive number")
        if not _is_number(max_ack_delay) or max_ack_delay < 0:
            raise ValueError("max_ack_delay must not be negative")
        if not _is_number(granularity) or granularity <= 0:
            raise ValueError("granularity must be a positive number")
        if not _is_number(min_rto) or min_rto <= 0 or not _is_number(max_rto) or max_rto < min_rto:
            raise ValueError("the rto bounds must be positive and ordered")
        self.initial_rtt = float(initial_rtt)
        self.max_ack_delay = float(max_ack_delay)
        self.granularity = float(granularity)
        self.min_rto = float(min_rto)
        self.max_rto = float(max_rto)
        self.latest_rtt = self.initial_rtt
        self.min_rtt: Optional[float] = None
        self.smoothed_rtt = self.initial_rtt
        self.rttvar = self.initial_rtt / 2.0
        self.samples = 0

    def update_rtt(self, latest_rtt: float, ack_delay: float = 0.0) -> float:
        """Fold one RTT sample into the estimator (RFC 9002 section 5.3)."""
        if not _is_number(latest_rtt) or latest_rtt <= 0:
            raise ValueError("latest_rtt must be a positive number")
        if not _is_number(ack_delay) or ack_delay < 0:
            raise ValueError("ack_delay must not be negative")
        latest_rtt = float(latest_rtt)
        delay = min(float(ack_delay), self.max_ack_delay)
        if self.samples == 0:
            self.min_rtt = latest_rtt
            self.smoothed_rtt = latest_rtt
            self.rttvar = latest_rtt / 2.0
        else:
            if latest_rtt < self.min_rtt + delay:
                self.min_rtt = latest_rtt
                adjusted = latest_rtt
            else:
                self.min_rtt = min(self.min_rtt, latest_rtt)
                adjusted = latest_rtt - delay
            self.rttvar = (1.0 - BETA) * self.rttvar + BETA * abs(self.smoothed_rtt - adjusted)
            self.smoothed_rtt += ALPHA * (adjusted - self.smoothed_rtt)
        self.latest_rtt = latest_rtt
        self.samples += 1
        return self.smoothed_rtt

    def pto_duration(self) -> float:
        """The probe timeout period derived from the smoothed estimate."""
        return (self.smoothed_rtt
                + max(4.0 * self.rttvar, self.granularity)
                + self.max_ack_delay)

    def rto(self) -> float:
        """The retransmission timeout derived from the smoothed estimate."""
        base = self.smoothed_rtt + max(4.0 * self.rttvar, self.granularity)
        return min(max(base, self.min_rto), self.max_rto)


class QuicConnection:
    """Sender side state: congestion window, in flight packets and timers."""

    def __init__(self, cwnd: int = INITIAL_WINDOW, ssthresh: float = float("inf"),
                 rtt: Optional[RttEstimator] = None,
                 max_ack_delay: float = MAX_ACK_DELAY) -> None:
        if not _is_number(cwnd) or cwnd <= 0:
            raise ValueError("cwnd must be a positive number")
        self.cwnd = int(cwnd)
        self.ssthresh = ssthresh
        self.rtt = rtt if rtt is not None else RttEstimator(max_ack_delay=max_ack_delay)
        self.sent_packets: Dict[int, SentPacket] = {}
        self.largest_sent_packet = -1
        self.largest_acked_packet: Optional[int] = None
        self.bytes_in_flight = 0
        self.recovery_start_time: Optional[float] = None
        self.pto_count = 0
        self.loss_time: Optional[float] = None
        self.loss_detection_deadline: Optional[float] = None
        self.lost_packets: List[int] = []

    def can_send(self) -> bool:
        """True while the congestion window has room for another packet."""
        return self.bytes_in_flight < self.cwnd

    def on_packet_sent(self, packet_number: int, now: float, size: int = MSS,
                       ack_eliciting: bool = True) -> SentPacket:
        """Register a packet that just left the sender at virtual time now."""
        if not isinstance(packet_number, int) or isinstance(packet_number, bool):
            raise InvalidPacketError("packet_number must be an integer")
        if packet_number <= self.largest_sent_packet:
            raise InvalidPacketError("packet_number must be larger than any packet sent before")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise InvalidPacketError("size must be a positive integer")
        now = _check_time(now)
        packet = SentPacket(packet_number, now, size, bool(ack_eliciting))
        self.sent_packets[packet_number] = packet
        self.largest_sent_packet = packet_number
        if packet.in_flight:
            self.bytes_in_flight += size
        return packet

    def on_ack_received(self, now: float, largest_acked: int, acked: Iterable[int],
                        ack_delay: float = 0.0) -> bool:
        """Process one acknowledgement frame; True when it changed the state."""
        now = _check_time(now)
        if not isinstance(largest_acked, int) or isinstance(largest_acked, bool) or largest_acked < 0:
            raise InvalidPacketError("largest_acked must be a non-negative integer")
        if largest_acked > self.largest_sent_packet:
            raise InvalidPacketError("acknowledgement for a packet that was never sent")
        if not _is_number(ack_delay) or ack_delay < 0:
            raise InvalidPacketError("ack_delay must not be negative")

        newly_acked: List[SentPacket] = []
        seen = set()
        for packet_number in acked:
            if not isinstance(packet_number, int) or isinstance(packet_number, bool):
                raise InvalidPacketError("acknowledged packet numbers must be integers")
            if packet_number in seen:
                continue
            seen.add(packet_number)
            packet = self.sent_packets.get(packet_number)
            if packet is not None:
                newly_acked.append(packet)
        newly_acked.sort(key=lambda packet: packet.packet_number)
        if not newly_acked:
            return False

        self._sample_rtt(now, largest_acked, newly_acked, ack_delay)

        acked_bytes = 0
        for packet in newly_acked:
            del self.sent_packets[packet.packet_number]
            if packet.in_flight:
                acked_bytes += packet.size
                self.bytes_in_flight -= packet.size

        if self.largest_acked_packet is None or largest_acked > self.largest_acked_packet:
            self.largest_acked_packet = largest_acked

        self.pto_count = 0

        if acked_bytes:
            self._on_packets_acked(acked_bytes)

        lost = self.detect_lost_packets(now)
        if lost:
            self._on_packets_lost(lost, now)

        self.set_loss_detection_timer(now)
        return True

    def _sample_rtt(self, now: float, largest_acked: int,
                    newly_acked: List[SentPacket], ack_delay: float) -> None:
        """Take an RTT sample from this frame when the estimate is trustworthy."""
        if not any(packet.ack_eliciting for packet in newly_acked):
            return
        for packet in newly_acked:
            if packet.packet_number == largest_acked:
                latest_rtt = now - packet.sent_time
                if latest_rtt > 0:
                    self.rtt.update_rtt(latest_rtt, ack_delay)
                return

    def _on_packets_acked(self, acked_bytes: int) -> None:
        """Grow the congestion window for the bytes that just left flight."""
        if self.cwnd < self.ssthresh:
            self.cwnd += acked_bytes
        else:
            self.cwnd += (acked_bytes * MSS) // self.cwnd

    def detect_lost_packets(self, now: float) -> List[SentPacket]:
        """Declare packets lost by the packet or the time threshold."""
        now = _check_time(now)
        self.loss_time = None
        if self.largest_acked_packet is None:
            return []
        loss_delay = TIME_THRESHOLD * max(self.rtt.latest_rtt, self.rtt.smoothed_rtt)
        loss_delay = max(loss_delay, self.rtt.granularity)
        lost: List[SentPacket] = []
        for packet_number in sorted(self.sent_packets):
            packet = self.sent_packets[packet_number]
            if (self.largest_acked_packet - packet_number >= K_PACKET_THRESHOLD
                    or packet.sent_time <= now - loss_delay):
                lost.append(packet)
            elif self.loss_time is None:
                self.loss_time = packet.sent_time + loss_delay
        for packet in lost:
            del self.sent_packets[packet.packet_number]
            if packet.in_flight:
                self.bytes_in_flight -= packet.size
            self.lost_packets.append(packet.packet_number)
        return lost

    def _on_packets_lost(self, lost: List[SentPacket], now: float) -> None:
        """Apply the congestion reaction for one batch of lost packets."""
        oldest_sent_time = min(packet.sent_time for packet in lost)
        if (self.recovery_start_time is not None
                and oldest_sent_time <= self.recovery_start_time):
            return
        self.recovery_start_time = now
        self.cwnd = max(int(self.cwnd * LOSS_REDUCTION_FACTOR), MINIMUM_WINDOW)

    def pto_period(self) -> float:
        """The probe timeout period used to arm the loss detection timer."""
        return (2 ** self.pto_count) * self.rtt.pto_duration()

    def set_loss_detection_timer(self, now: float) -> Optional[float]:
        """Arm the single timer that drives loss detection and probing."""
        now = _check_time(now)
        if self.bytes_in_flight == 0:
            self.loss_detection_deadline = None
        elif self.loss_time is not None:
            self.loss_detection_deadline = self.loss_time
        else:
            self.loss_detection_deadline = now + self.pto_period()
        return self.loss_detection_deadline

    def on_loss_detection_timeout(self, now: float) -> str:
        """Run whichever loss detection duty owns the timer that just expired."""
        now = _check_time(now)
        if self.loss_time is not None and self.loss_time <= now:
            self.loss_time = None
            lost = self.detect_lost_packets(now)
            if lost:
                self._on_packets_lost(lost, now)
            self.set_loss_detection_timer(now)
            return "loss_time"
        self.pto_count += 1
        self.set_loss_detection_timer(now)
        return "pto"
