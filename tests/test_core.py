"""Behavioural tests for the QUIC congestion control and loss recovery kernel."""

import unittest

from quic.core import (
    INITIAL_WINDOW,
    LOSS_REDUCTION_FACTOR,
    MAX_ACK_DELAY,
    MAX_RTO,
    MINIMUM_WINDOW,
    MIN_RTO,
    MSS,
    TIME_THRESHOLD,
    TIMER_GRANULARITY,
    InvalidPacketError,
    QuicConnection,
    RttEstimator,
)


class RttEstimatorTests(unittest.TestCase):
    def test_smoothed_rtt_and_variance_track_the_samples(self):
        """Samples are folded into the smoothed figure and the variance."""
        rtt = RttEstimator()
        rtt.update_rtt(0.100)
        self.assertEqual(rtt.samples, 1)
        self.assertAlmostEqual(rtt.latest_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.min_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.smoothed_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.rttvar, 0.050, places=9)

        rtt.update_rtt(0.200)
        self.assertEqual(rtt.samples, 2)
        self.assertAlmostEqual(rtt.latest_rtt, 0.200, places=9)
        self.assertAlmostEqual(rtt.min_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.smoothed_rtt, 0.1125, places=9)
        self.assertAlmostEqual(rtt.rttvar, 0.0625, places=9)

        rtt.update_rtt(0.100)
        self.assertEqual(rtt.samples, 3)
        self.assertAlmostEqual(rtt.min_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.smoothed_rtt, 0.1109375, places=9)
        self.assertAlmostEqual(rtt.rttvar, 0.050, places=9)
        self.assertGreaterEqual(rtt.smoothed_rtt, rtt.min_rtt)

    def test_reported_ack_delay_never_pushes_smoothed_rtt_below_min_rtt(self):
        """A peer reported ack delay is only credited when the sample covers it."""
        rtt = RttEstimator()
        rtt.update_rtt(0.100)
        rtt.update_rtt(0.100, ack_delay=MAX_ACK_DELAY)
        self.assertAlmostEqual(rtt.latest_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.min_rtt, 0.100, places=9)
        self.assertAlmostEqual(rtt.smoothed_rtt, 0.100, places=9)
        self.assertGreaterEqual(rtt.smoothed_rtt, rtt.min_rtt)

        before = rtt.smoothed_rtt
        rtt.update_rtt(0.500, ack_delay=MAX_ACK_DELAY)
        self.assertGreater(rtt.smoothed_rtt, before)
        self.assertLess(rtt.smoothed_rtt, 0.500)
        self.assertGreaterEqual(rtt.smoothed_rtt, rtt.min_rtt)

    def test_rto_stays_between_its_lower_and_upper_bound(self):
        """A short path must not drag the retransmission timeout under its floor."""
        rtt = RttEstimator()
        rtt.update_rtt(0.012)
        rtt.update_rtt(0.020)
        rtt.update_rtt(0.016)
        unclamped = rtt.smoothed_rtt + max(4.0 * rtt.rttvar, TIMER_GRANULARITY)
        self.assertLess(unclamped, MIN_RTO)
        self.assertAlmostEqual(rtt.rto(), MIN_RTO, places=9)
        self.assertGreaterEqual(rtt.rto(), MIN_RTO)
        self.assertLessEqual(rtt.rto(), MAX_RTO)

        rtt.update_rtt(5.000)
        self.assertGreater(rtt.rto(), MIN_RTO)
        self.assertLessEqual(rtt.rto(), MAX_RTO)
        self.assertAlmostEqual(
            rtt.rto(),
            rtt.smoothed_rtt + max(4.0 * rtt.rttvar, TIMER_GRANULARITY),
            places=9,
        )


class ProbeTimeoutTests(unittest.TestCase):
    def test_probe_period_grows_with_probes_and_resets_on_acknowledgement(self):
        """Consecutive probes back off, and a fresh acknowledgement clears the count."""
        conn = QuicConnection()
        conn.on_packet_sent(0, 0.000, size=MSS)
        self.assertTrue(conn.on_ack_received(0.200, 0, [0]))
        self.assertEqual(conn.rtt.samples, 1)
        self.assertAlmostEqual(conn.rtt.smoothed_rtt, 0.200, places=9)

        conn.on_packet_sent(1, 0.300, size=MSS)
        conn.on_packet_sent(2, 0.300, size=MSS)
        base = conn.rtt.pto_duration()
        self.assertAlmostEqual(conn.pto_period(), base, places=9)

        self.assertEqual(conn.on_loss_detection_timeout(0.500), "pto")
        self.assertEqual(conn.pto_count, 1)
        self.assertEqual(conn.on_loss_detection_timeout(1.000), "pto")
        self.assertEqual(conn.pto_count, 2)
        self.assertAlmostEqual(conn.pto_period(), 4 * base, places=9)

        self.assertTrue(conn.on_ack_received(1.500, 2, [2]))
        self.assertEqual(conn.pto_count, 0)
        self.assertAlmostEqual(conn.pto_period(), conn.rtt.pto_duration(), places=9)


class CongestionWindowTests(unittest.TestCase):
    def test_slow_start_grows_by_the_acknowledged_bytes(self):
        """Below the threshold the window grows by exactly the bytes acked."""
        conn = QuicConnection(cwnd=4 * MSS, ssthresh=64 * MSS)
        conn.on_packet_sent(0, 0.000, size=MSS)
        conn.on_packet_sent(1, 0.000, size=600)
        self.assertEqual(conn.bytes_in_flight, MSS + 600)
        self.assertTrue(conn.on_ack_received(0.100, 1, [0, 1]))
        self.assertEqual(conn.cwnd, 4 * MSS + MSS + 600)
        self.assertEqual(conn.bytes_in_flight, 0)

        conn2 = QuicConnection(cwnd=4 * MSS, ssthresh=64 * MSS)
        for packet_number in range(4):
            conn2.on_packet_sent(packet_number, 0.000, size=MSS)
        conn2.on_ack_received(0.100, 3, [0, 1, 2, 3])
        self.assertEqual(conn2.cwnd, 8 * MSS)
        self.assertLessEqual(conn2.cwnd, conn2.ssthresh)

        conn3 = QuicConnection(cwnd=4 * MSS, ssthresh=5 * MSS)
        for packet_number in range(4):
            conn3.on_packet_sent(packet_number, 0.000, size=MSS)
        conn3.on_ack_received(0.100, 3, [0, 1, 2, 3])
        self.assertEqual(conn3.cwnd, 8 * MSS)
        conn3.on_packet_sent(4, 0.200, size=MSS)
        conn3.on_ack_received(0.300, 4, [4])
        self.assertEqual(conn3.cwnd, 8 * MSS + max((MSS * MSS) // (8 * MSS), 1))

    def test_congestion_avoidance_grows_in_proportion_to_the_acked_bytes(self):
        """Above the threshold each acknowledgement adds a window scaled amount."""
        conn = QuicConnection(cwnd=10 * MSS, ssthresh=10 * MSS)
        for packet_number in range(3):
            conn.on_packet_sent(packet_number, 0.000, size=MSS)
        self.assertTrue(conn.on_ack_received(0.100, 2, [0, 1, 2]))
        self.assertEqual(conn.cwnd, 10 * MSS + (3 * MSS * MSS) // (10 * MSS))
        first_growth = conn.cwnd - 10 * MSS
        self.assertGreater(first_growth, 0)
        self.assertLess(first_growth, 3 * MSS)

        conn.on_packet_sent(3, 0.200, size=MSS)
        conn.on_ack_received(0.300, 3, [3])
        second_growth = conn.cwnd - (10 * MSS + first_growth)
        self.assertGreater(second_growth, 0)
        self.assertLess(second_growth, first_growth)


class LossDetectionTests(unittest.TestCase):
    def test_packet_threshold_marks_the_three_oldest_packets_lost(self):
        """A packet three numbers behind the largest ack is gone."""
        conn = QuicConnection(cwnd=10 * MSS, ssthresh=10 * MSS)
        for packet_number in range(6):
            conn.on_packet_sent(packet_number, 0.000, size=MSS)
        self.assertTrue(conn.on_ack_received(0.100, 5, [5]))
        self.assertEqual(conn.largest_acked_packet, 5)
        self.assertEqual(sorted(conn.lost_packets), [0, 1, 2])
        self.assertEqual(sorted(conn.sent_packets), [3, 4])

        conn.on_ack_received(0.200, 4, [4])
        self.assertEqual(sorted(conn.lost_packets), [0, 1, 2])

    def test_time_threshold_uses_the_larger_of_latest_and_smoothed_rtt(self):
        """A late packet waits out the longest recent round trip, not the average."""
        conn = QuicConnection(cwnd=10 * MSS, ssthresh=10 * MSS)
        conn.on_packet_sent(0, 0.000, size=MSS)
        conn.on_packet_sent(1, 0.000, size=MSS)
        self.assertTrue(conn.on_ack_received(0.100, 1, [1]))
        self.assertEqual(conn.rtt.samples, 1)
        self.assertAlmostEqual(conn.rtt.latest_rtt, 0.100, places=9)

        conn.on_packet_sent(2, 0.150, size=MSS)
        conn.on_packet_sent(3, 0.150, size=MSS)
        self.assertTrue(conn.on_ack_received(0.650, 3, [3]))
        self.assertAlmostEqual(conn.rtt.latest_rtt, 0.500, places=9)
        self.assertLess(conn.rtt.smoothed_rtt, conn.rtt.latest_rtt)

        self.assertEqual(conn.lost_packets, [0])
        self.assertEqual(sorted(conn.sent_packets), [2])
        self.assertIsNotNone(conn.loss_time)
        self.assertAlmostEqual(
            conn.loss_time,
            0.150 + TIME_THRESHOLD * conn.rtt.latest_rtt,
            places=9,
        )

    def test_one_loss_event_reduces_the_window_exactly_once(self):
        """Every packet of a single congestion event must not halve the window."""
        conn = QuicConnection(cwnd=10 * MSS, ssthresh=10 * MSS)
        for packet_number in range(5):
            conn.on_packet_sent(packet_number, 0.000, size=MSS)
        conn.on_packet_sent(5, 0.000, size=40, ack_eliciting=False)
        self.assertTrue(conn.on_ack_received(0.100, 5, [5]))

        self.assertEqual(sorted(conn.lost_packets), [0, 1, 2])
        self.assertEqual(
            conn.cwnd,
            max(int(10 * MSS * LOSS_REDUCTION_FACTOR), MINIMUM_WINDOW),
        )
        self.assertGreaterEqual(conn.cwnd, MINIMUM_WINDOW)
        self.assertLess(conn.cwnd, 10 * MSS)


class ValidationTests(unittest.TestCase):
    def test_invalid_input_and_stale_acknowledgements(self):
        """Out of range input is rejected and stale frames change nothing."""
        conn = QuicConnection(cwnd=10 * MSS, ssthresh=64 * MSS)
        with self.assertRaises(ValueError):
            conn.on_packet_sent(0, -0.001)
        with self.assertRaises(InvalidPacketError):
            conn.on_packet_sent(-1, 0.000)
        with self.assertRaises(InvalidPacketError):
            conn.on_packet_sent(0, 0.000, size=0)
        self.assertEqual(conn.largest_sent_packet, -1)
        self.assertEqual(conn.bytes_in_flight, 0)

        conn.on_packet_sent(0, 0.000, size=MSS)
        self.assertTrue(conn.on_ack_received(0.100, 0, [0]))
        samples = conn.rtt.samples
        cwnd = conn.cwnd
        self.assertEqual(samples, 1)
        self.assertFalse(conn.on_ack_received(0.200, 0, [0]))
        self.assertEqual(conn.rtt.samples, samples)
        self.assertEqual(conn.cwnd, cwnd)
        self.assertEqual(conn.bytes_in_flight, 0)
        self.assertIsNone(conn.loss_detection_deadline)

        conn.on_packet_sent(1, 0.300, size=40, ack_eliciting=False)
        self.assertEqual(conn.bytes_in_flight, 0)
        self.assertTrue(conn.on_ack_received(0.400, 1, [1]))
        self.assertEqual(conn.rtt.samples, samples)
        self.assertEqual(conn.cwnd, cwnd)

        with self.assertRaises(InvalidPacketError):
            conn.on_ack_received(0.500, 9, [9])
        with self.assertRaises(ValueError):
            conn.on_ack_received(-0.001, 1, [1])
        with self.assertRaises(InvalidPacketError):
            conn.on_ack_received(0.500, 1, [1], ack_delay=-0.001)


if __name__ == "__main__":
    unittest.main()
