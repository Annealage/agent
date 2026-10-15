"""``ReviewWatcher``: pushes ``review_changed`` when a product's review changes,
whoever changed it.

A review has writers the app does not control: the human editing the file by
hand, a ``git checkout``, a separately running agent writing a product's
published file directly (Mesh's ``mesh-callouts.json`` contract). So the
watcher samples the store's own ``state()``, a digest of what its files hold,
several times a second, and announces a change once it has settled. A change
made through the store itself is announced at once rather than on the next
sample: the store notifies its listeners after every write, and the watcher's
listener wakes the sampling loop.

The event carries no payload. The page refetches the review for itself
(``static/review.js`` through ``GET /review``, or a product's own route), so
that the review in the page has exactly one writer, the fetch; an event that
carried the content would be a second one.

The state machine is ``tick``, which takes the current time as a parameter
rather than reading a clock, and the sleeping loop is ``run``; splitting them
is what lets the deferral rule be tested by calling ``tick`` with the times a
test chooses, instead of by sleeping and hoping.
"""

import asyncio

from ..session.base import ReviewChanged

#: How often the store's files are sampled. One stat-and-read of a small JSON
#: file a quarter-second is cheaper than an inotify dependency and behaves the
#: same on every platform.
REVIEW_POLL_INTERVAL = 0.25

#: How long a change whose bytes never parse is waited on before it is
#: announced anyway.
REVIEW_MAX_DEFER = 5.0


class ReviewWatcher:
    """Publishes ``ReviewChanged`` through ``publish`` (the app's event
    publisher: event log, then broadcast) when ``store.state()`` changes."""

    def __init__(self, store, publish, interval=REVIEW_POLL_INTERVAL, max_defer=REVIEW_MAX_DEFER):
        self._store = store
        self._publish = publish
        self._interval = interval
        self._max_defer = max_defer
        # The first tick records what it finds and announces nothing: this
        # watcher reports changes since it started, and the state it starts in
        # is already covered by the page's own fetch on load. ``None`` is a
        # real announced value, meaning "nothing there", so a deletion is a
        # change; that is why this is a separate flag rather than a sentinel
        # in ``_announced``.
        self._primed = False
        self._announced = None
        self._unstable_since = None
        # Set by ``run`` for as long as it runs; ``poke`` is a no-op without
        # them, since there is no loop to wake.
        self._loop = None
        self._wake = None
        store.add_listener(self.poke)

    def poke(self):
        """Sample now rather than at the next interval. Safe to call from any
        thread, which matters because the store calls it from whichever
        executor thread made the write."""
        loop, wake = self._loop, self._wake
        if loop is None or wake is None:
            return
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            # The loop closed between the check and the call: shutdown, with
            # nobody left to announce anything to.
            pass

    async def tick(self, now):
        """Sample the store once and publish if it changed and looks settled.

        Returns True if an event was published. A change whose bytes do not
        yet parse is waited on rather than announced, but only up to
        ``max_defer``: a file being rewritten continuously never looks
        settled, and a watcher that waits for quiet that never comes is a
        watcher that never fires. Past that bound the event goes out anyway,
        which is safe because a page tolerates a fetch that fails and the next
        change produces another event.
        """
        loop = asyncio.get_running_loop()
        try:
            digest, settled = await loop.run_in_executor(None, self._store.state)
        except OSError:
            # A read that fails outright (a permission change, a directory
            # replacing the file) is left for the next tick rather than
            # treated as a change: there is nothing to tell the page to
            # refetch, and the fetch would report the same failure itself.
            return False
        if not self._primed:
            self._primed = True
            self._announced = digest
            return False
        if digest == self._announced:
            self._unstable_since = None
            return False
        if not settled:
            if self._unstable_since is None:
                self._unstable_since = now
            if now - self._unstable_since < self._max_defer:
                return False
        self._announced = digest
        self._unstable_since = None
        self._publish(ReviewChanged())
        return True

    async def run(self):
        """Sample on the interval, or at once when poked, until cancelled.

        The priming sample is taken before the first wait, not after it, so a
        change made in the first interval of the run is a change from the
        primed state rather than part of it.
        """
        loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._loop = loop
        try:
            await self.tick(loop.time())
            while True:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                await self.tick(loop.time())
        finally:
            self._loop = None
            self._wake = None
