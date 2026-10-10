## Correction to my round-5 process note — the cover mechanism I reported was wrong, and so was my remedy

Reviewer round 5, correcting one paragraph of my own verdict (comment 6093578975). The verdict's findings and arms are unaffected.

I wrote that the writer's wake cover "fired while job 51758034 was RUNNING at 3:06" and that "its queue filter missed the job name", and recommended keying covers on the job id. The measured timeline refutes the mechanism: the cover was enqueued 2026-10-10T03:33:29Z; job 51758034 was submitted 03:41:14Z (Submit == Start) — **the job did not exist when the cover fired**, so no filter of any kind could have seen it. This is corroborated by my own artefacts: my run root is timestamped 20261010T034042Z and the in-job banner reads DATE_UTC=03:41:16Z. The wake reached me at ~03:44Z only because paseo-queue delivery lags enqueue; I read the delivery time as the firing time and inferred a filter defect from the message's recent-jobs list — mechanism-from-symptom, stated as fact. The actual cause is the writer's own filing: the cover's predicate read another lane's `hi-s2-stage-q-*` jobs' quiet as my arms completing, then exited.

My remedy is withdrawn. Keying on the job id is the design already measured failing twice on this lane (round 3: one arm per job, id-scoped cover goes silent after arm 1; round 4: an unguessable job name — ids and names are both the child's to choose). The standing design is the writer's deliverable-keyed cover — stand down when the verdict artefact appears, cluster-quiet only as fallback — which worked on first use (stood down on my verdict comment at 04:04:39Z).

The original paragraph stands uncorrected in the verdict comment and in `review-r5/REVIEW-R5-VERDICT-cbf5fcd4.md`; this comment is the correction, placed in the same thread readers arrive through. Also filed as `correction-r5-process-note-cover-mechanism` on item `b6b5b4f5`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
