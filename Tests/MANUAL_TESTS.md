# macOS regression checklist

Run `swift test`, `swift build -c release`, and the README Xcode build command.
The XCTest suite tests editor completion state, collision-free export names,
clip-range validation, and GPU matching using generated textures. GPU tests may
skip on machines without Metal; a skipped GPU test is not a hardware validation.

## Editor interactions

1. Import clips A and B. Set independent trim ranges and tracking boxes; switch
   between them and confirm each green box is visible with its original size.
2. Drag a box farther than its own width. It should move without jumping to the
   pointer or turning into a new selection. Repeat at a zoom other than 100%.
3. Scrub away from the trimmed clip's first frame and draw a box. Processing must
   seek to the trim start and request a new selection, without starting export.
4. Draw the new box at that frame and export. While A processes, select B. A's
   completion must not replace B's preview, trim range, selection, or analysis.
5. Export a trimmed clip twice and clips with the same basename from different
   source folders. Check that all existing exports remain unchanged and that the
   editor stays on the original source duration/coordinates.
6. Change trim boundaries after drawing a box. If the new start differs from the
   selected frame, processing must request a new selection at the new start.

## Video processing

- Use landscape, portrait/90-degree metadata, 180-degree metadata and mirrored
  clips with the same visible feature. Check that tracking and exported display
  orientation agree with the selected feature.
- Test MP4, MOV and ProRes export, with/without audio, and fractional trim points.
  Check first/last frames, clip duration, audio synchronization and frame count.
- Test a corrupt source and an unwritable/full output volume. The operation must
  report failure, not a completed clip; partial output must not remain usable as
  a successful export.

`Scripts/run_synthetic_regression.py` is an optional older media benchmark. It
requires ffmpeg/ffprobe, Xcode, and the four fixture videos listed by the script
under `/tmp/stableflow_regression2`; those fixtures are not in the repository.
It does not replace the self-contained XCTest suite or the interactive checks.
