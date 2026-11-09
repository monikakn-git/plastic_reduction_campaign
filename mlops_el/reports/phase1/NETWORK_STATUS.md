# Acquisition connectivity interruption

**Resolved on resumption:** the host and native MPS access were verified. Acquisition completed with 10,000 distinct originals and no failed transfers after two exact duplicate copies were replaced with unused annotated source images. See `acquisition.json` and the versioned selection manifest for the current facts. The interruption history below is retained as evidence.

The project target remains **10,000 original pLitterStreet images**. The last main-pass checkpoint reported 2,836 verified images and 7,164 failed transfers. The dominant error was DNS resolution failure; timeouts and certificate hostname errors were also reported. No certificate checks were disabled and failed transfers were not accepted as valid images.

The full-data baseline has not started. The acquisition gate prevents training on this partial dataset.

Two attempts to stop the running downloader/workflow were not executed because automatic permission review first timed out and then disconnected. The review explicitly stated that this was a review-service failure, not a determination that the stop action was unsafe. At that point, the earlier process could still be retrying because it had not loaded the newly added outage-stop safeguard. The subsequent process check found no stale workflow running.

Before restarting, stop the old run and restore connectivity. The updated acquisition code stops further network requests after a DNS/TLS failure, preserves downloaded originals, and can resume from verified files. Resume must use the existing selected dataset and must not lower the 10,000-image target.
