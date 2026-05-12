#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_NAME="StableFlow Pro"
APP_DIR="$DIST_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
EXECUTABLE_NAME="StableFlow"
PRODUCT_BUNDLE_IDENTIFIER="com.stableflow.pro"
MARKETING_VERSION="${MARKETING_VERSION:-0.1.0}"
CURRENT_PROJECT_VERSION="${CURRENT_PROJECT_VERSION:-1}"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
ARM64_BUILD_DIR="$ROOT_DIR/.build/arm64-apple-macosx/release"
X86_64_BUILD_DIR="$ROOT_DIR/.build/x86_64-apple-macosx/release"
UNIVERSAL_BINARY="$DIST_DIR/$EXECUTABLE_NAME-universal"

mkdir -p "$DIST_DIR"

echo "==> Building arm64 release executable"
swift build -c release --arch arm64 --package-path "$ROOT_DIR"

echo "==> Building x86_64 release executable"
swift build -c release --arch x86_64 --package-path "$ROOT_DIR"

echo "==> Creating universal executable"
rm -f "$UNIVERSAL_BINARY"
lipo -create \
  "$ARM64_BUILD_DIR/$EXECUTABLE_NAME" \
  "$X86_64_BUILD_DIR/$EXECUTABLE_NAME" \
  -output "$UNIVERSAL_BINARY"

echo "==> Creating app bundle"
rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cp "$UNIVERSAL_BINARY" "$MACOS_DIR/$EXECUTABLE_NAME"
cp -R "$ARM64_BUILD_DIR/StableFlow_StableFlow.bundle" "$RESOURCES_DIR/StableFlow_StableFlow.bundle"

# 复制应用图标
if [[ -f "$ROOT_DIR/Resources/AppIcon.icns" ]]; then
  cp "$ROOT_DIR/Resources/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
  echo "==> App icon copied"
fi

PLIST_PATH="$CONTENTS_DIR/Info.plist"
cp "$ROOT_DIR/Resources/Info.plist" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleName $APP_NAME" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $APP_NAME" "$PLIST_PATH" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :CFBundleDisplayName string $APP_NAME" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $PRODUCT_BUNDLE_IDENTIFIER" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleExecutable $EXECUTABLE_NAME" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $MARKETING_VERSION" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $CURRENT_PROJECT_VERSION" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $MACOSX_DEPLOYMENT_TARGET" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleDevelopmentRegion en" "$PLIST_PATH"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile AppIcon" "$PLIST_PATH" 2>/dev/null || /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string AppIcon" "$PLIST_PATH"

echo "==> Ad-hoc signing app bundle"
if ! codesign --force --deep --sign - "$APP_DIR"; then
  echo "==> Warning: codesign failed, leaving app unsigned for local use"
fi

echo "==> Binary architectures"
lipo -info "$MACOS_DIR/$EXECUTABLE_NAME"

echo "App bundle ready:"
echo "$APP_DIR"
