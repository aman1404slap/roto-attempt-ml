1. Parsed all 6 Silhouette projects across 4 versions: 6,969 splines / 52,565 keyframes / 660 frames. Shapes are B-splines, not Beziers as the reference docs assumed - so our output has to be B-spline native, that is what the artist edits.
2. Reimplemented the Silhouette rendering conventions; achieved 0.9843–0.9980 matte overlap, with 0.9947 on the main full-res test. Already told this to Jon, he is okay with it. Side finding: 2 of 6 shots have a colour transform wrongly applied to the alpha at the vendor's end (50% coverage stored as 0.214) - we invert it on read, but their pipeline should be told
3. Recovered the layer → RGB channel mapping through brute-force search. But if we have actual mapping present, it would be helpful. Because out of 23 channels, we were able to recover 9 reliably
4. Validated our training representation: artist file → arrays → file → render = 1.000000 overlap on all 9 usable elements. essentially this tells us that our training data is correct and usable
5. Packaged 9 trustworthy elements / 1,200 frames / 1,310 shapes / 10,957 keyframes / 23 MB.
6. Ran a small end-to-end model successfully; predicted programs render back at 0.9984–1.0000.
7. This is essentially an overfitting which is our current plan that gives us a good starting point.
8. Keyframe recall is good (0.75–1.00), but precision is poor (0.21–0.44) → model is over-keying by ~2.5×.
9. One dataset (sh0230) has a ~20-frame mismatch between archived project and delivered matte. Already told varun. The sfx does not have the data that exr has, most likely sfx is older and exr was generated on newer which was not given.

10. Decided motion blur, feather and single-frame paint-stroke hair are out as POC targets, in as fields we carry through faithfully - they are render settings, not spline structure, and the hair strokes have 1 key by construction so they teach key-every-frame. This was Lokesh's question; the handoff's Phase 0 already skips both.
11. MAT_0130 has our best render score (0.998) but the worst keyframe economy in the set (1.67 keys per live frame vs 0.03–0.64) - keeping it out of sparsity training.

Next Steps:
1. Improve baseline primarily by increasing keyframe precision.
2. Add non-keyframe render-consistency loss

Question/Need from jon/varun

1. Layer → channel mapping, especially sh0260 — could unlock 10 more elements.
2. Newer sh0230 save (sfx).
3. Silhouette license needed to validate the final editable deliverable. (maybe we can skip this for now).
4. RGB plates for a subset - not blocking us now (Tier 1 does not need them), but needed for the RGB branch and Tier 2.


