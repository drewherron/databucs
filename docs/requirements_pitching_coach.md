# Requirements: series prep report (pitching coach / advance scout)

*Written from the point of view of the people who use the report. It sets what the
dashboard must answer before any of it is built.*

## Who uses it, and when

- **The pitching coach and the advance scout** use it the day before a series and the
  morning of each game. They plan how each pitcher attacks each hitter, then brief the
  pitchers and catchers.
- **Pitchers and catchers** see the parts that concern them, in meetings or on a
  printout. Neither group reads raw numbers well under time pressure.

## Questions it must answer

1. **For this hitter and this pitcher, what should we throw in each situation?** Cover
   the first pitch, ahead in the count (0-2, 1-2), behind (2-0, 3-1), full count, needing
   a strikeout (runner on third, less than two outs) and two outs with a runner in
   scoring position.
2. **Is that different from what the pitcher already does?** Advice that matches his
   habit is a confirmation; advice that differs is a change worth a conversation. The
   report should make the difference obvious.
3. **How confident is it?** Tell me when options are basically equal, so I don't drill a
   pitcher on a choice that doesn't matter.
4. **Where is this hitter weak?** A picture of the zone shows that faster than a table.
5. **Which of our pitchers match up best against this hitter?** This is an overview
   across the staff, not a ranking to the decimal.

## What makes it usable

- **Plain language.** Use pitch names ("slider", not "SL") and locations relative to the
  hitter ("low and away", "up and in, off the plate"), not zone numbers or the catcher's
  left and right.
- **Tiers, not decimals.** Group options as *strong*, *fine* and *avoid*. Show the
  numbers for whoever wants them, but don't rank options that are within noise of each
  other.
- **Few clicks.** Pick the series, then the hitter, then the pitcher. Everything on the
  page follows those choices.
- **Honest about its limits.** Say on the page that it assumes league-average command,
  doesn't account for sequencing, and is built from pitches pitchers chose to throw.
- **Current before every series.** Run it with the probable starters and a projected
  lineup, and allow a custom lineup when the real one is posted.

## Out of scope for this version

- Matchup rankings with plate-appearance projections (strikeout and walk rates, damage)
  and bullpen availability
- Similar hitters for batters with little data, and how predictable a pitch would be
- Late-and-close and times-through-the-order situations
- A printable per-batter report (a later step; this version is the dashboard)

## Done when

Someone who hasn't seen the project can open the dashboard and do three things without
help:
- pick the series, a Detroit hitter and a Pirates pitcher
- say what to throw at 0-2 and at 3-1, and whether that is a change from the pitcher's
  habit
- see which Pirates pitchers match up best with that hitter
