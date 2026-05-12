# StableFlow Pro

StableFlow Pro is a native macOS video stabilization app focused on one workflow: manually select a visual feature, track it, and export a stabilized clip that keeps the selected feature centered.

The app is built with SwiftUI, AVFoundation, Vision, Core Image, and Metal. It runs locally on macOS and does not use a web UI, cloud processing, account login, or online licensing.

## What It Does

- Imports one or more video files through file picker, drag and drop, or macOS `open`.
- Lets the user draw one tracking box around the feature to lock.
- Tracks the selected feature with a Vision/GPU/template-assisted pipeline.
- Centers the tracked feature during export instead of doing generic whole-frame stabilization.
- Supports clip trimming before export.
- Exports MP4, MOV, or ProRes depending on the selected format.
- Preserves source audio when possible.
- Includes Mac App Store-oriented sandbox/export configuration.

## Design Goal

StableFlow Pro is intended for shots where a specific object detail should stay visually locked, such as a car badge, logo, plate area, product detail, or other high-contrast feature.

It deliberately avoids non-core workflows such as batch processing, side-by-side comparison, multi-point tracking, account systems, and cloud services. The goal is a simple local tool that does one thing well.

## Requirements

- macOS 13.0 or later
- Xcode 15 or later
- Swift 5.9 or later
- Apple Silicon or Intel Mac

## Build

Build the Swift package:

```bash
swift build -c release
```

Build the native Xcode app:

```bash
xcodebuild \
  -project StableFlowPro.xcodeproj \
  -scheme StableFlowPro \
  -configuration Release \
  -destination "generic/platform=macOS" \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Create a local app bundle:

```bash
Scripts/build_macos_app.sh
```

Create a local DMG after building the app bundle:

```bash
Scripts/create_dmg.sh
```

## Mac App Store Build

The repository includes a Mac App Store build helper and sandbox entitlements:

```bash
Scripts/build_mac_app_store.sh
```

To export or upload to App Store Connect, you still need your own Apple Developer account, Team ID, signing certificates, and App Store Connect app record.

## Project Structure

- `Sources/ContentView.swift`: SwiftUI app UI and processing workflow.
- `Sources/ImprovedStabilizer.swift`: feature tracking, trajectory filtering, centering transforms, rendering, and audio muxing.
- `Sources/GPUStabilizer.swift`: GPU-assisted tracking utilities.
- `Sources/MetalStabilizer.swift`: Metal rendering and stabilization support.
- `Sources/Shaders.metal`: Metal shader code.
- `Resources/`: app icon, Info.plist, entitlements, and export options.
- `Scripts/`: build, DMG, App Store, and regression helper scripts.
- `SPEC.md`: product scope and functional specification.

## Notes

This repository contains an original Swift implementation and does not include third-party proprietary application code, proprietary assets, account bypasses, or license bypass logic.

## License

MIT License. See `LICENSE`.
