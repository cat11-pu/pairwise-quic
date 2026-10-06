# quic

A dependency free QUIC congestion control and loss recovery kernel.

The package models the sending half of a QUIC connection without touching a
real socket. Time and acknowledgements are injected by the caller one step at
a time, so a whole transfer is deterministic and can be replayed from a test.
It covers the RTT estimator with ACK delay compensation, the RTO and PTO
figures with exponential back off, slow start, congestion avoidance, packet
and time threshold loss detection and the recovery period that guards the
congestion window against repeated reductions.

## Layout

    quic/__init__.py     public names re-exported by the package
    quic/core.py         congestion control and loss recovery kernel
    tests/__init__.py    test package marker
    tests/test_core.py   behavioural test suite

## Running the tests

From the project root:

    python3 -m unittest discover -s tests -v

Only the Python standard library is required; there is nothing to install.
