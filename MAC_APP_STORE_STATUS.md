## Mac App Store Status

Last verified: 2026-04-23

### Current state

The project has been adapted for a Mac App Store build:

- App Sandbox entitlements are enabled.
- Export output uses a user-selected folder with security-scoped bookmarks.
- The Xcode target uses a fixed `Info.plist` and App Store entitlements.
- A native Xcode Release build succeeds.
- An unsigned `.xcarchive` succeeds.

### Files added or changed for App Store work

- `Resources/StableFlowPro-AppStore.entitlements`
- `Resources/AppStoreExportOptions.plist`
- `Resources/Info.plist`
- `Sources/ContentView.swift`
- `StableFlowPro.xcodeproj/project.pbxproj`
- `Scripts/build_mac_app_store.sh`

### Verified commands

Release build:

```bash
xcodebuild -project "StableFlowPro.xcodeproj" \
  -scheme StableFlowPro \
  -configuration Release \
  -destination "generic/platform=macOS" \
  CODE_SIGNING_ALLOWED=NO build
```

Unsigned archive:

```bash
xcodebuild -project "StableFlowPro.xcodeproj" \
  -scheme StableFlowPro \
  -configuration Release \
  -destination "generic/platform=macOS" \
  CODE_SIGNING_ALLOWED=NO archive \
  -archivePath "dist/StableFlowPro-AppStore-unsigned.xcarchive"
```

Both succeeded on 2026-04-23.

### Current blockers

#### 1. No signing identity on this Mac

This currently returns zero identities:

```bash
security find-identity -v -p codesigning
```

Without a valid App Store signing identity, the project cannot be exported or uploaded to App Store Connect.

Expected identities are typically one of:

- `Apple Distribution`
- any App Store-capable signing identity available through the Apple Developer team configured for this app

#### 2. Team configuration is still empty

The Xcode build settings currently report:

```text
_DEVELOPMENT_TEAM_IS_EMPTY = YES
```

That means the project is not yet attached to an Apple Developer team for signing/export.

#### 3. Export options still contain a placeholder team ID

`Resources/AppStoreExportOptions.plist` still uses a placeholder:

```text
YOURTEAMID
```

This must be replaced with the real Apple Developer Team ID before export/upload.

### Archive status

Unsigned archive path:

`dist/StableFlowPro-AppStore-unsigned.xcarchive`

Bundle identifier inside archive:

`com.stableflow.pro`

### What must happen next

1. Sign into the correct Apple Developer account in Xcode.
2. Install or unlock a valid App Store signing identity on this Mac.
3. Set the correct Development Team in the Xcode target.
4. Replace `YOURTEAMID` in `Resources/AppStoreExportOptions.plist`.
5. Run `Scripts/build_mac_app_store.sh`.
6. Upload the exported archive to App Store Connect.
7. Complete App Store Connect metadata, privacy answers, screenshots, pricing, and review notes.

### Notes

- Metal Toolchain was previously a blocker and has already been installed.
- The remaining blockers are Apple account, signing, and App Store Connect configuration, not source-level build failures.
