# TASK-ver3 — the asks behind P7, P11 and P12, verbatim from the session (2026-09-11 / 12)

These stages were requested in chat, not in TASK.md / TASK-ver2.md. Recorded here so the "asked for X, ran Y"
pre-flight can be done on them (verification finding 5, 2026-09-12). Quotes are the user's messages as sent.

## P7 (review re-check of the Δp_null headline)

Session opened with the review brief:

> I have a research claim and the code and raw data behind it. I did not write this code and I have no stake in the
> claim being true — I need to know if it's wrong before I put it in a writeup that gets read carefully. Don't verify
> that the code does what it says. Verify that the number means what it's being taken to mean. […] 1. Read the script
> and tell me what it actually computes […] 2. Recompute the headline number from the raw output file independently.
> 3. Then: what is the dumbest way this result could be wrong?

followed by:

> so hypothesis partially correct for score probe>e>label probe? what it entail if label probe the worst. also fix
> the but the pre-registered Δp_null headline cannot distinguish any of the three scorers at this n, and the "E 15
> vs probe 11" sentence is comparing across different honest pools — on the same pool it's 11 vs 11. I'd rewrite the
> claim around the AUROC CIs and state the Δp_null test as inconclusive rather than falsified.

## P11 (pool-wide probe-beats-E audit)

> so one of the row beat E although there is the explanation above. can you search for more like this for my
> writeup? create graph as well to explain it

(context: the P8 paragraph on `30e71655#26`, the one quiet payload where a probe beat E, and the user's own note
that text 99 / dumb 100 / integer 8 reach the same row.)

## P12 (forced readout after a truncated CoT, trajectory 30e71655)

Case-study request:

> can you do specific case on 30e71655 how e soce and i score and probe as well. can you see why I score so big
> compared to E? do u have access in reasoning as well? i assume it score so big because its complex redis file
> stuff that need multi level prep

then, quoting the proposed test back:

> go The test that would separate "can't" from "doesn't": force the readout after a truncated CoT — after hop 1 only
> ("removes the clear"), after hops 1+2 ("not needed for main task") — and see which hop moves the digit. That's a
> forced-readout-at-intermediate-positions experiment on this trajectory; the activations for it aren't captured, so
> it needs the model. P5 territory, your call.

Matched-register control (added 2026-09-12 from the verification section's "main missing control"):

> *P12: no length- or register-matched append.* Every lift-producing append is the same 51-token objective hop or
> its 45-token prefix; the only short appends are the 6-token verdict sentence and the 14-token benign clause.
> Nothing tests ~50 tokens of confident, technical, non-objective text, so "the readout follows a stated objective
> match" is not yet separated from "the readout follows any sustained suspicious-sounding elaboration".

## Not asked

P5 (from TASK-ver2 / the user's "p5 is next work") has not been started.
