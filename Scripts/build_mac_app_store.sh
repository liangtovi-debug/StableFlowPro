#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PATH="$ROOT_DIR/StableFlowPro.xcodeproj"
SCHEME="StableFlowPro"
CONFIGURATION="${CONFIGURATION:-Release}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$ROOT_DIR/dist/StableFlowPro-AppStore.xcarchive}"
EXPORT_PATH="${EXPORT_PATH:-$ROOT_DIR/dist/appstore-export}"
EXPORT_OPTIONS_PLIST="${EXPORT_OPTIONS_PLIST:-$ROOT_DIR/Resources/AppStoreExportOptions.plist}"

if ! xcodebuild -version >/dev/null 2>&1; then
  echo "xcodebuild not found"
  exit 1
fi

if ! security find-identity -v -p codesigning | grep -q "3rd Party Mac Developer Application\\|Apple Distribution"; then
  echo "Missing App Store signing identity."
  echo "Install an App Store distribution certificate before archiving for Mac App Store."
  exit 1
fi

if /usr/libexec/PlistBuddy -c "Print :teamID" "$EXPORT_OPTIONS_PLIST" 2>/dev/null | grep -q "YOURTEAMID"; then
  echo "Missing real Apple Developer Team ID."
  echo "Replace YOURTEAMID in $EXPORT_OPTIONS_PLIST before exporting for App Store Connect."
  exit 1
fi

rm -rf "$ARCHIVE_PATH" "$EXPORT_PATH"

echo "==> Archiving Mac App Store build"
xcodebuild \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination "generic/platform=macOS" \
  archive \
  -archivePath "$ARCHIVE_PATH"

echo "==> Exporting App Store package"
xcodebuild \
  -exportArchive \
  -archivePath "$ARCHIVE_PATH" \
  -exportPath "$EXPORT_PATH" \
  -exportOptionsPlist "$EXPORT_OPTIONS_PLIST"

echo "Archive ready:"
echo "$ARCHIVE_PATH"
echo
echo "Exported package:"
echo "$EXPORT_PATH"
