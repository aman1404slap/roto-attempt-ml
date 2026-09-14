# Plain English

Five explanations, no jargon. Read them in order.

1. **[dataset.md](dataset.md)** — what arrived in the 50 folders, and what we picked out of it
2. **[s0.md](s0.md)** — the first real attempt: what goes in, how it learns, what we got
3. **[s1.md](s1.md)** — the second attempt, which mostly didn't work, and why that was still useful
4. **[s2.md](s2.md)** — the third: taking away what we tell it about shape lifespans and point counts
5. **[s3.md](s3.md)** — the fourth, and the big one: taking away *which shape is which*

Each stage that involves training keeps a **diary** — every attempt, what it cost, what it
bought, and whether it was on the plan or a retry, with a picture each:

- **[s2-attempts.md](s2-attempts.md)** — S2, three attempts, all passed, zero retries
- **[s3-attempts.md](s3-attempts.md)** — S3, in progress

They are separate files because the two stages are read against different baselines: S2 asks
whether the picture *stays* where it was, S3 asks whether there is a picture at all.

There is a second folder, `steps/`, with the same material written for an engineer. Same facts,
harder reading. This folder is the one to read first.
