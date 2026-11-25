# Autosampler

Drop in a song, get a playable sampler: tempo-locked chops on your Mac keyboard, an auto-arranged loop, and drums that follow the song's own groove.

## Run

```
./run.sh
```

Opens http://localhost:8765. Needs Python 3.11+ and ffmpeg. Songs live in `Vinyls/`.

## Keys

| | |
|---|---|
| `Q W E R T Y U I` / `A S D F G H J K` | pads |
| `Space` | play / stop |
| `Enter` | record into the loop |
| `Tab` | Keys mode: play the selected pad chromatically (`Z` / `X` octave) |
| `←` `→` | move the selected pad to the previous / next hit (`Shift` = fine) |
| `⌘Z` | undo |

Drag the BPM, Tune, Level and Swing numbers; double-click resets them.

## How it works

- **Stems**: Demucs splits the song; you choose what to sample before analysis.
- **Tempo**: Beat This activations, a comb-refined period, and a tempo-locked dynamic-programming beat tracker; the grid is then aligned to real transients. 6/8 is detected and handled.
- **Slices**: onsets are walked back to the start of the attack so the hit lands on the keypress; pitch comes from pYIN on the relevant stem.
- **Drums**: kick/snare/hat hits and the bar pattern are extracted from the song's drum stem, giving a "Song" kit and groove; synthesized kits are also included.
