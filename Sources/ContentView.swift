import SwiftUI
import AVKit
import AVFoundation
import CoreImage
import Metal
import MetalKit
import AppKit
import UniformTypeIdentifiers

private let supportedVideoExtensions: Set<String> = ["mp4", "mov", "m4v", "avi", "mkv", "mts", "m2ts", "wmv"]
private let outputFolderBookmarkDefaultsKey = "StableFlowOutputFolderBookmark"

extension Notification.Name {
    static let stableFlowImportFiles = Notification.Name("StableFlowImportFiles")
}

private func isSupportedVideoURL(_ url: URL) -> Bool {
    supportedVideoExtensions.contains(url.pathExtension.lowercased())
}

private func supportedVideoContentTypes() -> [UTType] {
    supportedVideoExtensions.compactMap { UTType(filenameExtension: $0) }
}

final class StableFlowAppDelegate: NSObject, NSApplicationDelegate {
    func application(_ application: NSApplication, open urls: [URL]) {
        let filtered = urls.filter(isSupportedVideoURL(_:))
        guard !filtered.isEmpty else { return }
        NotificationCenter.default.post(
            name: .stableFlowImportFiles,
            object: nil,
            userInfo: ["urls": filtered]
        )
    }
}

// MARK: - AVPlayerView 包装器 (替代 SwiftUI VideoPlayer)
struct AVPlayerViewRepresentable: NSViewRepresentable {
    let player: AVPlayer
    
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .none
        view.showsFullScreenToggleButton = false
        return view
    }
    
    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        nsView.player = player
    }
}

struct PreviewZoomEventLayer: NSViewRepresentable {
    let onScrollZoom: (CGFloat) -> Void
    let onMagnify: (CGFloat) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> PreviewZoomNSView {
        let view = PreviewZoomNSView()
        context.coordinator.attach(to: view)
        context.coordinator.onScrollZoom = onScrollZoom
        context.coordinator.onMagnify = onMagnify
        return view
    }

    func updateNSView(_ nsView: PreviewZoomNSView, context: Context) {
        context.coordinator.onScrollZoom = onScrollZoom
        context.coordinator.onMagnify = onMagnify
    }

    static func dismantleNSView(_ nsView: PreviewZoomNSView, coordinator: Coordinator) {
        coordinator.detach()
    }

    final class Coordinator {
        fileprivate var onScrollZoom: ((CGFloat) -> Void)?
        fileprivate var onMagnify: ((CGFloat) -> Void)?
        private weak var view: PreviewZoomNSView?
        private var scrollMonitor: Any?
        private var magnifyMonitor: Any?

        func attach(to view: PreviewZoomNSView) {
            self.view = view
            if scrollMonitor == nil {
                scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                    guard let self, self.contains(event: event) else { return event }
                    self.onScrollZoom?(-event.scrollingDeltaY)
                    return nil
                }
            }
            if magnifyMonitor == nil {
                magnifyMonitor = NSEvent.addLocalMonitorForEvents(matching: .magnify) { [weak self] event in
                    guard let self, self.contains(event: event) else { return event }
                    self.onMagnify?(event.magnification)
                    return nil
                }
            }
        }

        func detach() {
            if let scrollMonitor {
                NSEvent.removeMonitor(scrollMonitor)
                self.scrollMonitor = nil
            }
            if let magnifyMonitor {
                NSEvent.removeMonitor(magnifyMonitor)
                self.magnifyMonitor = nil
            }
        }

        private func contains(event: NSEvent) -> Bool {
            guard let view, let window = view.window, event.window === window else { return false }
            let point = view.convert(event.locationInWindow, from: nil)
            return view.bounds.contains(point)
        }
    }
}

final class PreviewZoomNSView: NSView {
    override var isOpaque: Bool { false }

    override func hitTest(_ point: NSPoint) -> NSView? {
        nil
    }
}

struct PlaybackShortcutLayer: NSViewRepresentable {
    let onTogglePlay: () -> Void
    let onStepBackward: () -> Void
    let onStepForward: () -> Void
    let onSetClipStart: () -> Void
    let onSetClipEnd: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> ShortcutPassthroughView {
        let view = ShortcutPassthroughView()
        context.coordinator.attach(
            togglePlay: onTogglePlay,
            stepBackward: onStepBackward,
            stepForward: onStepForward,
            setClipStart: onSetClipStart,
            setClipEnd: onSetClipEnd
        )
        return view
    }

    func updateNSView(_ nsView: ShortcutPassthroughView, context: Context) {
        context.coordinator.attach(
            togglePlay: onTogglePlay,
            stepBackward: onStepBackward,
            stepForward: onStepForward,
            setClipStart: onSetClipStart,
            setClipEnd: onSetClipEnd
        )
    }

    static func dismantleNSView(_ nsView: ShortcutPassthroughView, coordinator: Coordinator) {
        coordinator.detach()
    }

    final class Coordinator {
        private var keyMonitor: Any?
        private var onTogglePlay: (() -> Void)?
        private var onStepBackward: (() -> Void)?
        private var onStepForward: (() -> Void)?
        private var onSetClipStart: (() -> Void)?
        private var onSetClipEnd: (() -> Void)?

        func attach(
            togglePlay: @escaping () -> Void,
            stepBackward: @escaping () -> Void,
            stepForward: @escaping () -> Void,
            setClipStart: @escaping () -> Void,
            setClipEnd: @escaping () -> Void
        ) {
            onTogglePlay = togglePlay
            onStepBackward = stepBackward
            onStepForward = stepForward
            onSetClipStart = setClipStart
            onSetClipEnd = setClipEnd

            guard keyMonitor == nil else { return }
            keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self, NSApp.isActive else { return event }
                if NSApp.keyWindow?.firstResponder is NSTextView {
                    return event
                }

                switch event.keyCode {
                case 49:
                    self.onTogglePlay?()
                    return nil
                case 123:
                    self.onStepBackward?()
                    return nil
                case 124:
                    self.onStepForward?()
                    return nil
                default:
                    break
                }

                let key = event.charactersIgnoringModifiers?.lowercased()
                if key == "k" {
                    self.onTogglePlay?()
                    return nil
                } else if key == "j" {
                    self.onStepBackward?()
                    return nil
                } else if key == "l" {
                    self.onStepForward?()
                    return nil
                } else if key == "i" || key == "[" {
                    self.onSetClipStart?()
                    return nil
                } else if key == "o" || key == "]" {
                    self.onSetClipEnd?()
                    return nil
                }
                return event
            }
        }

        func detach() {
            if let keyMonitor {
                NSEvent.removeMonitor(keyMonitor)
                self.keyMonitor = nil
            }
        }
    }
}

final class ShortcutPassthroughView: NSView {
    override var isOpaque: Bool { false }

    override func hitTest(_ point: NSPoint) -> NSView? {
        nil
    }
}

enum StabilizeQuality: String, CaseIterable {
    case fast = "快速"
    case balanced = "平衡"
    case best = "最佳"
}

enum ExportFormat: String, CaseIterable {
    case mp4 = "MP4"
    case mov = "MOV"
    case prores = "ProRes"

    var fileExtension: String {
        switch self {
        case .mp4:
            return "mp4"
        case .mov, .prores:
            return "mov"
        }
    }

    var stabilizerValue: String {
        switch self {
        case .mp4:
            return "mp4"
        case .mov:
            return "mov"
        case .prores:
            return "prores"
        }
    }
}

@main
struct StableFlowPro: App {
    @NSApplicationDelegateAdaptor(StableFlowAppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("StableFlow Pro", id: "main") {
            MainView()
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1200, height: 800)
    }
}

// MARK: - 主视图
struct MainView: View {
    @StateObject private var vm = ProVideoVM()
    
    var body: some View {
        VStack(spacing: 0) {
            // 顶部工具栏
            ToolbarView(vm: vm)
                .frame(height: 44)
            
            HSplitView {
                // 左侧：当前素材
                ProjectPanel(vm: vm)
                    .frame(width: 180)
                
                // 中间：预览（大）
                PreviewPanel(vm: vm)
                    .frame(minWidth: 600)
            }
            
            // 底部：属性面板
            InspectorPanel(vm: vm)
                .frame(height: 260)
        }
        .background(
            PlaybackShortcutLayer(
                onTogglePlay: { vm.togglePlay() },
                onStepBackward: { vm.seekRelative(by: -1.0) },
                onStepForward: { vm.seekRelative(by: 1.0) },
                onSetClipStart: { vm.setClipStartToCurrentTime() },
                onSetClipEnd: { vm.setClipEndToCurrentTime() }
            )
        )
        .onReceive(NotificationCenter.default.publisher(for: .stableFlowImportFiles)) { notification in
            guard let urls = notification.userInfo?["urls"] as? [URL] else { return }
            vm.importExternalFiles(urls)
        }
    }
}

// MARK: - 项目面板
struct ProjectPanel: View {
    @ObservedObject var vm: ProVideoVM
    @State private var isDropTargeted = false
    
    var body: some View {
        VStack(spacing: 0) {
            // 标题
            HStack {
                Text("素材")
                    .font(.headline)
                Spacer()
                Button(action: { vm.importVideo() }) {
                    Image(systemName: "plus")
                }
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            
            Divider()
            
            if vm.clips.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "film.stack")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("拖拽视频到此处")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("导入视频") {
                        vm.importVideo()
                    }
                    .buttonStyle(.bordered)
                }
                .frame(maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(vm.clips.indices, id: \.self) { idx in
                            ClipRow(
                                clip: vm.clips[idx],
                                isSelected: idx == vm.selectedIndex,
                                onTap: { vm.selectClip(idx) }
                            )
                        }
                    }
                    .padding(8)
                }

                Text("可多选导入；稳定处理只处理当前选中素材")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)
            }
        }
        .background(
            Group {
                if isDropTargeted {
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                        .padding(6)
                }
            }
        )
        .onDrop(of: [UTType.fileURL.identifier], isTargeted: $isDropTargeted) { providers in
            vm.importProviders(providers)
        }
    }
}

struct ClipRow: View {
    let clip: VideoClip
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 8) {
                Image(systemName: "film")
                    .foregroundColor(isSelected ? .white : .secondary)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(clip.name)
                        .font(.caption)
                        .foregroundColor(isSelected ? .white : .primary)
                        .lineLimit(1)
                    Text(clip.duration)
                        .font(.caption2)
                        .foregroundColor(isSelected ? .white.opacity(0.7) : .secondary)
                }
                
                Spacer()
                
                if clip.isProcessed {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.caption)
                }
            }
            .padding(8)
            .background(isSelected ? Color.accentColor : Color.clear)
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - 预览面板
struct PreviewPanel: View {
    @ObservedObject var vm: ProVideoVM
    @State private var isDropTargeted = false
    
    var body: some View {
        VStack(spacing: 0) {
            // 预览区（不带工具栏，避免重复）
            GeometryReader { geo in
                ZStack {
                    if let player = vm.player {
                        let videoRect = vm.videoDisplayRect(in: geo.size)

                        AVPlayerViewRepresentable(player: player)
                            .frame(width: videoRect.width, height: videoRect.height)
                            .position(x: videoRect.midX, y: videoRect.midY)
                            .clipped()

                        TrackerOverlay(vm: vm, containerSize: geo.size)

                        PreviewZoomEventLayer(
                            onScrollZoom: { delta in
                                vm.adjustZoom(by: delta)
                            },
                            onMagnify: { magnification in
                                vm.adjustZoomByMagnification(magnification)
                            }
                        )
                    } else {
                        VStack(spacing: 16) {
                            Image(systemName: "film")
                                .font(.system(size: 64))
                                .foregroundColor(.secondary)
                            Text("导入视频开始")
                                .font(.title3)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black)
            .overlay {
                if isDropTargeted {
                    ZStack {
                        Color.black.opacity(0.35)
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [10, 8]))
                            .padding(24)
                        VStack(spacing: 10) {
                            Image(systemName: "tray.and.arrow.down.fill")
                                .font(.system(size: 44))
                            Text("松开即可导入视频")
                                .font(.headline)
                        }
                        .foregroundColor(.white)
                    }
                }
            }
            .onDrop(of: [UTType.fileURL.identifier], isTargeted: $isDropTargeted) { providers in
                vm.importProviders(providers)
            }
            
            // 播放控制
            PlaybackBar(vm: vm)
        }
    }
}

// MARK: - 工具栏
struct ToolbarView: View {
    @ObservedObject var vm: ProVideoVM
    
    var body: some View {
        HStack(spacing: 16) {
            // 稳定控制
            if vm.player != nil {
                HStack(spacing: 8) {
                    Text("缩放 \(Int(vm.zoom * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()

            // 处理状态
            if vm.isProcessing {
                HStack(spacing: 6) {
                    ProgressView().scaleEffect(0.6)
                    Text(vm.processStatus)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .frame(maxWidth: 180, alignment: .leading)
                }
            }

            // 导出按钮
            if vm.selectedClip?.isProcessed == true {
                Button(action: { vm.revealProcessedVideo() }) {
                    Label("查看导出文件", systemImage: "magnifyingglass")
                }
                .buttonStyle(.bordered)
            }

            Button(action: { _ = vm.chooseOutputFolder() }) {
                Label("输出目录", systemImage: "folder")
            }
            .buttonStyle(.bordered)

            // 单个处理按钮
            Button(action: { vm.processVideo() }) {
                Label("稳定处理", systemImage: "wand.and.stars")
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
            .disabled(vm.selectedClip == nil || vm.trackerPoints.isEmpty || vm.isProcessing)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - 跟踪点覆盖层（单点跟踪）
struct TrackerOverlay: View {
    @ObservedObject var vm: ProVideoVM
    var containerSize: CGSize = .zero
    
    @State private var isSelecting = false
    @State private var selectionStart = CGPoint.zero
    @State private var selectionEnd = CGPoint.zero
    @State private var trackerDragOrigin: CGPoint?

    private func clampedPoint(_ point: CGPoint, to rect: CGRect) -> CGPoint {
        CGPoint(
            x: min(max(point.x, rect.minX), rect.maxX),
            y: min(max(point.y, rect.minY), rect.maxY)
        )
    }

    private func currentTrackerRect(in videoRect: CGRect, videoWidth: CGFloat, videoHeight: CGFloat) -> CGRect? {
        guard let idx = vm.selectedTrackerIndex, idx < vm.trackerPoints.count else { return nil }
        let point = vm.trackerPoints[idx]
        let viewX = videoRect.minX + (point.position.x / videoWidth) * videoRect.width
        let viewY = videoRect.minY + (point.position.y / videoHeight) * videoRect.height
        let boxWidth = videoRect.width * (point.boxSize.width / videoWidth)
        let boxHeight = videoRect.height * (point.boxSize.height / videoHeight)
        return CGRect(
            x: viewX - boxWidth / 2,
            y: viewY - boxHeight / 2,
            width: boxWidth,
            height: boxHeight
        )
    }
    
    var body: some View {
        GeometryReader { geo in
            let vw = vm.videoWidth > 0 ? vm.videoWidth : 1920
            let vh = vm.videoHeight > 0 ? vm.videoHeight : 1080
            let videoRect = vm.videoDisplayRect(in: geo.size)
            
            ZStack {
                let centerX = videoRect.midX
                let centerY = videoRect.midY
                
                Path { p in
                    p.move(to: CGPoint(x: centerX - 14, y: centerY))
                    p.addLine(to: CGPoint(x: centerX + 14, y: centerY))
                    p.move(to: CGPoint(x: centerX, y: centerY - 14))
                    p.addLine(to: CGPoint(x: centerX, y: centerY + 14))
                }
                .stroke(Color.white.opacity(0.85), style: StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                
                if vm.showDebug, let analysis = vm.stabilizationAnalysis {
                    if analysis.rawTrajectory.count > 1 {
                        Path { p in
                            for (index, point) in analysis.rawTrajectory.enumerated() {
                                let x = videoRect.minX + point.x * videoRect.width
                                let y = videoRect.minY + point.y * videoRect.height
                                if index == 0 {
                                    p.move(to: CGPoint(x: x, y: y))
                                } else {
                                    p.addLine(to: CGPoint(x: x, y: y))
                                }
                            }
                        }
                        .stroke(Color.orange.opacity(0.7), lineWidth: 1.5)
                    }
                    
                    if analysis.smoothedTrajectory.count > 1 {
                        Path { p in
                            for (index, point) in analysis.smoothedTrajectory.enumerated() {
                                let x = videoRect.minX + point.x * videoRect.width
                                let y = videoRect.minY + point.y * videoRect.height
                                if index == 0 {
                                    p.move(to: CGPoint(x: x, y: y))
                                } else {
                                    p.addLine(to: CGPoint(x: x, y: y))
                                }
                            }
                        }
                        .stroke(Color.cyan.opacity(0.9), lineWidth: 2)
                    }

                    if analysis.renderedTrajectory.count > 1 {
                        Path { p in
                            for (index, point) in analysis.renderedTrajectory.enumerated() {
                                let x = videoRect.minX + point.x * videoRect.width
                                let y = videoRect.minY + point.y * videoRect.height
                                if index == 0 {
                                    p.move(to: CGPoint(x: x, y: y))
                                } else {
                                    p.addLine(to: CGPoint(x: x, y: y))
                                }
                            }
                        }
                        .stroke(Color.green.opacity(0.9), style: StrokeStyle(lineWidth: 2.5, dash: [6, 4]))
                    }
                }
                
                // 只显示一个跟踪点（选中的）
                if let idx = vm.selectedTrackerIndex, idx < vm.trackerPoints.count {
                    let point = vm.trackerPoints[idx]
                    let viewX = videoRect.minX + (point.position.x / vw) * videoRect.width
                    let viewY = videoRect.minY + (point.position.y / vh) * videoRect.height
                    let boxWidth = videoRect.width * (point.boxSize.width / vw)
                    let boxHeight = videoRect.height * (point.boxSize.height / vh)
                    
                    ZStack {
                        Rectangle()
                            .fill(Color.clear)
                            .frame(width: boxWidth, height: boxHeight)
                            .contentShape(Rectangle())
                            .position(x: viewX, y: viewY)

                        // 跟踪框
                        Rectangle()
                            .stroke(Color.green, lineWidth: 3)
                            .frame(width: boxWidth, height: boxHeight)
                            .position(x: viewX, y: viewY)
                            .allowsHitTesting(false)
                        
                        // 十字
                        Path { p in
                            p.move(to: CGPoint(x: viewX - 15, y: viewY))
                            p.addLine(to: CGPoint(x: viewX + 15, y: viewY))
                            p.move(to: CGPoint(x: viewX, y: viewY - 15))
                            p.addLine(to: CGPoint(x: viewX, y: viewY + 15))
                        }
                        .stroke(Color.green, lineWidth: 2)
                        .allowsHitTesting(false)
                    }
                }
                
                // 框选区域
                if isSelecting {
                    let start = clampedPoint(selectionStart, to: videoRect)
                    let end = clampedPoint(selectionEnd, to: videoRect)
                    let rect = CGRect(
                        x: min(start.x, end.x),
                        y: min(start.y, end.y),
                        width: abs(end.x - start.x),
                        height: abs(end.y - start.y)
                    )
                    Rectangle()
                        .stroke(Color.yellow, lineWidth: 2)
                        .background(Color.yellow.opacity(0.2))
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .simultaneousGesture(
                DragGesture(minimumDistance: 5)
                    .onChanged { v in
                        if !isSelecting, trackerDragOrigin == nil {
                            vm.pauseForTrackingSelection()
                        }
                        if !isSelecting, trackerDragOrigin == nil,
                           let trackerRect = currentTrackerRect(
                               in: videoRect,
                               videoWidth: CGFloat(vw),
                               videoHeight: CGFloat(vh)
                           ), trackerRect.insetBy(dx: -8, dy: -8).contains(v.startLocation) {
                            trackerDragOrigin = CGPoint(x: trackerRect.midX, y: trackerRect.midY)
                        }
                        if let origin = trackerDragOrigin, let idx = vm.selectedTrackerIndex {
                            vm.moveTrackerPoint(idx, to: CGPoint(
                                x: origin.x + v.translation.width,
                                y: origin.y + v.translation.height
                            ), in: geo.size)
                            return
                        }
                        let start = clampedPoint(v.startLocation, to: videoRect)
                        let current = clampedPoint(v.location, to: videoRect)
                        if !isSelecting {
                            isSelecting = true
                            selectionStart = start
                            selectionEnd = current
                        } else {
                            selectionEnd = current
                        }
                    }
                    .onEnded { v in
                        let wasSelecting = isSelecting
                        isSelecting = false
                        if trackerDragOrigin != nil {
                            trackerDragOrigin = nil
                            return
                        }
                        guard wasSelecting else { return }
                        let start = clampedPoint(v.startLocation, to: videoRect)
                        let end = clampedPoint(v.location, to: videoRect)
                        let rect = CGRect(
                            x: min(start.x, end.x),
                            y: min(start.y, end.y),
                            width: abs(end.x - start.x),
                            height: abs(end.y - start.y)
                        )
                        guard rect.width >= 6, rect.height >= 6 else { return }

                        let centerPoint = CGPoint(x: rect.midX, y: rect.midY)
                        guard videoRect.contains(centerPoint) else { return }
                        
                        let normalized = vm.normalizedVideoPoint(from: centerPoint, in: geo.size)
                        let vx = normalized.x * vw
                        let vy = normalized.y * vh
                        let boxWidth = rect.width / videoRect.width * vw
                        let boxHeight = rect.height / videoRect.height * vh
                        
                        // 只保留一个跟踪点
                        vm.trackerPoints = [TrackerPoint(
                            position: CGPoint(x: vx, y: vy),
                            boxSize: CGSize(
                                width: max(boxWidth, 24),
                                height: max(boxHeight, 24)
                            ),
                            referenceTime: vm.trackingSelectionTime
                        )]
                        vm.stabilizationAnalysis = nil
                        vm.selectedTrackerIndex = 0
                    }
            )
        }
        .allowsHitTesting(vm.isVideoReady)
    }
}

// MARK: - 播放控制条
struct PlaybackBar: View {
    @ObservedObject var vm: ProVideoVM
    
    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 16) {
                // 时间显示
                Text(vm.currentTimeString)
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 60)
                
                // 播放控制
                HStack(spacing: 8) {
                    Button(action: { vm.seekStart() }) {
                        Image(systemName: "backward.end.fill")
                    }
                    
                    Button(action: { vm.togglePlay() }) {
                        Image(systemName: vm.isPlaying ? "pause.fill" : "play.fill")
                            .font(.title3)
                    }
                    
                    Button(action: { vm.seekEnd() }) {
                        Image(systemName: "forward.end.fill")
                    }
                }
                .buttonStyle(.plain)

                TimelineView(vm: vm)
                    .frame(height: 44)

                Text(vm.durationString)
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 60)
            }

            HStack(spacing: 10) {
                Button("设为片段开始") {
                    vm.setClipStartToCurrentTime()
                }
                .disabled(vm.player == nil)

                Button("设为片段结束") {
                    vm.setClipEndToCurrentTime()
                }
                .disabled(vm.player == nil)

                Button("重置片段") {
                    vm.resetClipRange()
                }
                .disabled(vm.player == nil)

                Spacer()

                Text(vm.clipRangeText)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - 时间轴视图
struct TimelineView: View {
    @ObservedObject var vm: ProVideoVM
    @State private var activeDragTarget: TimelineDragTarget?
    @State private var dragTrimStartOrigin: Double?
    @State private var dragTrimEndOrigin: Double?
    @State private var dragSelectionStartOrigin: Double?
    @State private var dragSelectionEndOrigin: Double?

    private enum TimelineDragTarget {
        case startHandle
        case endHandle
        case selection
        case seekBar
    }
    
    var body: some View {
        GeometryReader { geo in
            let startX = geo.size.width * vm.clipStart
            let endX = geo.size.width * vm.clipEnd
            let handleHitWidth: CGFloat = 28
            let handleVisibleWidth: CGFloat = 10
            let clampedStartX = min(max(startX, handleHitWidth / 2), geo.size.width - handleHitWidth / 2)
            let clampedEndX = min(max(endX, handleHitWidth / 2), geo.size.width - handleHitWidth / 2)
            let selectionWidth = max(0, clampedEndX - clampedStartX)
            ZStack(alignment: .leading) {
                // 背景
                Rectangle()
                    .fill(Color(nsColor: .textBackgroundColor).opacity(0.65))

                WaveformRepresentation(vm: vm, size: geo.size)
                    .allowsHitTesting(false)

                // 处理进度叠加——替代 Inspector 里的独立进度条
                if vm.isProcessing {
                    Rectangle()
                        .fill(Color.green.opacity(0.35))
                        .frame(width: geo.size.width * vm.processProgress)
                        .animation(.linear(duration: 0.15), value: vm.processProgress)
                        .allowsHitTesting(false)
                } else if vm.selectedClip?.isProcessed == true {
                    Rectangle()
                        .fill(Color.green.opacity(0.35))
                        .allowsHitTesting(false)
                }

                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        SpatialTapGesture()
                            .onEnded { value in
                                let tapX = value.location.x
                                let isTrimmed = vm.clipStart > 0.001 || vm.clipEnd < 0.999
                                if isTrimmed {
                                    let onStartHandle = abs(tapX - clampedStartX) <= handleHitWidth / 2
                                    let onEndHandle = abs(tapX - clampedEndX) <= handleHitWidth / 2
                                    let onSelection = tapX >= clampedStartX + handleVisibleWidth
                                        && tapX <= clampedEndX - handleVisibleWidth
                                    guard !onStartHandle, !onEndHandle, !onSelection else { return }
                                }
                                let pos = max(0, min(1, tapX / max(geo.size.width, 1)))
                                vm.seek(to: pos)
                            }
                    )
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 1)
                            .onChanged { value in
                                let pos = max(0, min(1, value.location.x / max(geo.size.width, 1)))
                                vm.seek(to: pos)
                            }
                    )
                
                // (波形图在14px进度条下不显示)
                
                Rectangle()
                    .fill(Color.red.opacity(vm.clipStart > 0.001 || vm.clipEnd < 0.999 ? 0.28 : 0.12))
                    .frame(width: geo.size.width * vm.clipWidth)
                    .offset(x: geo.size.width * vm.clipStart)
                    .allowsHitTesting(false)

                Rectangle()
                    .stroke(Color.red.opacity(vm.clipStart > 0.001 || vm.clipEnd < 0.999 ? 0.75 : 0.35), lineWidth: 1.5)
                    .frame(width: geo.size.width * vm.clipWidth)
                    .offset(x: geo.size.width * vm.clipStart)
                    .allowsHitTesting(false)

                if selectionWidth > handleVisibleWidth * 2 && vm.clipWidth < 0.999 {
                    Color.clear
                        .frame(width: selectionWidth - handleVisibleWidth * 2, height: geo.size.height)
                        .position(
                            x: clampedStartX + selectionWidth / 2,
                            y: geo.size.height / 2
                        )
                        .contentShape(Rectangle())
                        .highPriorityGesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    activeDragTarget = .selection
                                    if dragSelectionStartOrigin == nil || dragSelectionEndOrigin == nil {
                                        dragSelectionStartOrigin = vm.clipStart
                                        dragSelectionEndOrigin = vm.clipEnd
                                    }
                                    let delta = value.translation.width / max(geo.size.width, 1)
                                    let originStart = dragSelectionStartOrigin ?? vm.clipStart
                                    let originEnd = dragSelectionEndOrigin ?? vm.clipEnd
                                    let clipWidth = originEnd - originStart
                                    let newStart = max(0, min(1 - clipWidth, originStart + delta))
                                    vm.clipStart = newStart
                                    vm.clipEnd = min(1, newStart + clipWidth)
                                }
                                .onEnded { _ in
                                    dragSelectionStartOrigin = nil
                                    dragSelectionEndOrigin = nil
                                    activeDragTarget = nil
                                }
                        )
                        .zIndex(1)
                }

                Color.clear
                    .frame(width: handleHitWidth, height: geo.size.height)
                    .position(x: clampedStartX, y: geo.size.height / 2)
                    .contentShape(Rectangle())
                    .highPriorityGesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                activeDragTarget = .startHandle
                                if dragTrimStartOrigin == nil {
                                    dragTrimStartOrigin = vm.clipStart
                                }
                                let delta = value.translation.width / max(geo.size.width, 1)
                                let origin = dragTrimStartOrigin ?? vm.clipStart
                                let newStart = max(0, min(vm.clipEnd - 0.05, origin + delta))
                                vm.clipStart = newStart
                            }
                            .onEnded { _ in
                                dragTrimStartOrigin = nil
                                activeDragTarget = nil
                            }
                    )
                    .zIndex(2)

                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.red)
                    .frame(width: handleVisibleWidth, height: geo.size.height)
                    .position(x: clampedStartX, y: geo.size.height / 2)
                    .allowsHitTesting(false)
                    .zIndex(3)

                Color.clear
                    .frame(width: handleHitWidth, height: geo.size.height)
                    .position(x: clampedEndX, y: geo.size.height / 2)
                    .contentShape(Rectangle())
                    .highPriorityGesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                activeDragTarget = .endHandle
                                if dragTrimEndOrigin == nil {
                                    dragTrimEndOrigin = vm.clipEnd
                                }
                                let delta = value.translation.width / max(geo.size.width, 1)
                                let origin = dragTrimEndOrigin ?? vm.clipEnd
                                let newEnd = min(1, max(vm.clipStart + 0.05, origin + delta))
                                vm.clipEnd = newEnd
                            }
                            .onEnded { _ in
                                dragTrimEndOrigin = nil
                                activeDragTarget = nil
                            }
                    )
                    .zIndex(2)

                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.red)
                    .frame(width: handleVisibleWidth, height: geo.size.height)
                    .position(x: clampedEndX, y: geo.size.height / 2)
                    .allowsHitTesting(false)
                    .zIndex(3)

                // 播放头
                Rectangle()
                    .fill(Color.blue)
                    .frame(width: 2, height: geo.size.height)
                    .offset(x: geo.size.width * vm.progress)
                    .allowsHitTesting(false)

                // (标签已移除——14px 进度条不需要文字叠加)
            }
            .cornerRadius(4)
        }
    }
}

// MARK: - 波形图表示
struct WaveformRepresentation: View {
    @ObservedObject var vm: ProVideoVM
    let size: CGSize
    
    var body: some View {
        Canvas { context, canvasSize in
            let barCount = 100
            let barWidth = canvasSize.width / CGFloat(barCount)
            
            for i in 0..<barCount {
                let waveSeed = sin(Double(i) * 0.19) * 0.5 + cos(Double(i) * 0.07) * 0.35
                let normalized = max(0.12, min(1.0, 0.5 + waveSeed))
                let height = CGFloat(normalized) * (canvasSize.height - 18)
                let x = CGFloat(i) * barWidth
                let y = (canvasSize.height - height) / 2
                
                let rect = CGRect(x: x, y: y, width: barWidth - 1, height: height)
                let inSelection = Double(i) / Double(max(barCount - 1, 1)) >= vm.clipStart
                    && Double(i) / Double(max(barCount - 1, 1)) <= vm.clipEnd
                context.fill(
                    Path(rect),
                    with: .color(inSelection ? Color.accentColor.opacity(0.4) : Color.gray.opacity(0.28))
                )
            }
        }
    }
}

// MARK: - 属性面板
struct InspectorPanel: View {
    @ObservedObject var vm: ProVideoVM
    
    var body: some View {
        VStack(spacing: 0) {
            Text("属性")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
            
            Divider()
            
            ScrollView {
                VStack(spacing: 20) {
                    // 稳定设置
                    if vm.selectedClip != nil {
                        GroupBox {
                            VStack(alignment: .leading, spacing: 16) {
                                // 质量
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("处理质量")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    
                                    Picker("", selection: $vm.quality) {
                                        ForEach(StabilizeQuality.allCases, id: \.self) { q in
                                            Text(q.rawValue).tag(q)
                                        }
                                    }
                                    .pickerStyle(.segmented)
                                }
                                
                                // 平滑度
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("平滑度")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                        Spacer()
                                        Text("\(Int(vm.smoothness * 100))%")
                                            .font(.caption)
                                    }
                                    Slider(value: $vm.smoothness, in: 0...1)
                                }
                                
                                // 稳定强度 (新功能)
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("稳定强度")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                        Spacer()
                                        Text("\(Int(vm.stabilizeStrength * 100))%")
                                            .font(.caption)
                                    }
                                    Slider(value: $vm.stabilizeStrength, in: 0...1)
                                    Text(vm.trackerPoints.isEmpty ? "无跟踪框时控制普通防抖补偿。" : "有跟踪框时同时控制特征居中的跟随收敛。")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                
                                // 滚动快门修复
                                Toggle("滚动快门修复", isOn: $vm.rollingShutter)
                                    .font(.caption)
                                
                                // 导出格式
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("导出格式")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    
                                    Picker("", selection: $vm.exportFormat) {
                                        ForEach(ExportFormat.allCases, id: \.self) { format in
                                            Text(format.rawValue).tag(format)
                                        }
                                    }
                                    .pickerStyle(.segmented)
                                }
                            }
                            .padding(8)
                        }
                        
                        // 处理链信息
                        GroupBox {
                            VStack(alignment: .leading, spacing: 8) {
                                if !vm.trackerPoints.isEmpty {
                                    Text("当前：特征跟踪居中模式")
                                        .font(.caption2)
                                        .foregroundColor(.blue)
                                } else {
                                    Text("当前：请在预览区拖拽框选要锁定的特征")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding(8)
                        }
                        
                        // 视频信息
                        GroupBox {
                            VStack(alignment: .leading, spacing: 8) {
                                if let clip = vm.selectedClip {
                                    InfoRow(label: "分辨率", value: clip.resolution)
                                    InfoRow(label: "帧率", value: clip.frameRate)
                                    InfoRow(label: "编码", value: clip.codec)
                                    InfoRow(label: "大小", value: clip.fileSize)
                                    InfoRow(label: "时长", value: clip.duration)
                                } else {
                                    Text("无视频")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding(8)
                        }

                        if let analysis = vm.stabilizationAnalysis {
                            GroupBox {
                                VStack(alignment: .leading, spacing: 8) {
                                    InfoRow(label: "模式", value: analysis.isCentering ? "特征居中" : "普通防抖")
                                    InfoRow(label: "平均中心偏差", value: String(format: "%.2f%%", analysis.meanCenterError * 100))
                                    InfoRow(label: "最大中心偏差", value: String(format: "%.2f%%", analysis.maxCenterError * 100))
                                    InfoRow(label: "平均补偿跳变", value: String(format: "%.2f px", analysis.meanTransformDelta))
                                    InfoRow(label: "最大补偿跳变", value: String(format: "%.2f px", analysis.maxTransformDelta))
                                    InfoRow(label: "自动裁切比例", value: String(format: "%.1f%%", analysis.cropRatio * 100))
                                    InfoRow(label: "裁切受限", value: analysis.cropLimited ? "是" : "否")
                                }
                                .padding(8)
                            }
                        }
                        
                        // 跟踪点
                        GroupBox {
                            VStack(alignment: .leading, spacing: 8) {
                                if vm.trackerPoints.isEmpty {
                                    Text("在预览区框选以添加跟踪点")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                } else {
                                    ForEach(vm.trackerPoints.indices, id: \.self) { idx in
                                        HStack {
                                            Text("跟踪点 \(idx + 1)")
                                                .font(.caption)
                                            Spacer()
                                            Button(action: { vm.removeTrackerPoint(idx) }) {
                                                Image(systemName: "xmark.circle.fill")
                                                    .foregroundColor(.red)
                                            }
                                        }
                                    }
                                }
                            }
                            .padding(8)
                        }
                    }
                    
                    // (处理状态和导出按钮已移至工具栏)
                }
                .padding()
            }
        }
    }
}

// MARK: - 信息行组件
struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

// MARK: - 数据模型
struct VideoClip: Identifiable {
    let id = UUID()
    let url: URL
    let name: String
    var duration: String  // 改为 var
    var isSecurityScoped: Bool = false
    var isProcessed: Bool = false
    var processedURL: URL?

    // 视频元数据
    var resolution: String = ""
    var frameRate: String = ""
    var codec: String = ""
    var fileSize: String = ""

    // 每条视频独立的跟踪框，切换时保留各自的设置
    var trackerPoints: [TrackerPoint] = []
    var clipStart: Double = 0
    var clipEnd: Double = 1
}

struct TrackerPoint: Identifiable {
    let id = UUID()
    var position: CGPoint
    var boxSize: CGSize = CGSize(width: 80, height: 80)
    var isTracking: Bool = true
    // Position belongs to this source frame, not necessarily the current playhead.
    var referenceTime: Double = 0
}

struct ProcessingConfig {
    let trackingPoint: CGPoint?
    let trackingBoxSize: CGSize
    let exportFormat: String
    let processingQuality: String
    let smoothness: Double
    let stabilizeStrength: Double
    let rollingShutterCompensation: Bool
    let clipStart: Double
    let clipEnd: Double
    let debugMode: Bool
}

struct LoadedVideoInfo {
    let duration: CMTime
    let naturalSize: CGSize
    let preferredTransform: CGAffineTransform
    let displaySize: CGSize
    let nominalFrameRate: Float
}

// MARK: - 运动数据（用于视频稳定）
struct MotionFrameData {
    var translationX: Double = 0      // X方向位移
    var translationY: Double = 0      // Y方向位移
    var rotation: Double = 0          // 旋转角度（弧度）
    var scale: Double = 1.0           // 缩放
    var confidence: Double = 1.0       // 置信度
}

// MARK: - 专业版ViewModel
@MainActor
class ProVideoVM: ObservableObject {
    @Published var clips: [VideoClip] = []
    @Published var selectedIndex: Int = -1
    @Published var player: AVPlayer?
    @Published var isPlaying: Bool = false
    @Published var progress: Double = 0
    @Published var currentTimeString: String = "00:00:00"
    @Published var durationString: String = "00:00:00"
    
    @Published var zoom: Double = 1.0
    @Published var trackerPoints: [TrackerPoint] = [] {
        didSet { persistCurrentClipEditingState() }
    }
    @Published var selectedTrackerIndex: Int? = nil  // 选中的跟踪点索引
    
    // 视频尺寸
    @Published var videoWidth: Double = 1920
    @Published var videoHeight: Double = 1080
    @Published var rawVideoWidth: Double = 1920
    @Published var rawVideoHeight: Double = 1080
    
    @Published var quality: StabilizeQuality = .balanced
    @Published var smoothness: Double = 0.5
    @Published var stabilizeStrength: Double = 0.8  // 稳定强度
    @Published var rollingShutter: Bool = true
    @Published var exportFormat: ExportFormat = .mp4
    
    // 调试选项
    @Published var showDebug: Bool = false  // 显示调试信息
    
    @Published var clipStart: Double = 0 {
        didSet { persistCurrentClipEditingState() }
    }
    @Published var clipEnd: Double = 1 {
        didSet { persistCurrentClipEditingState() }
    }
    
    @Published var isProcessing: Bool = false
    @Published var processProgress: Double = 0
    @Published var processStatus: String = "就绪"
    @Published var stabilizationAnalysis: StabilizationAnalysis?
    
    var selectedClip: VideoClip? {
        guard selectedIndex >= 0 && selectedIndex < clips.count else { return nil }
        return clips[selectedIndex]
    }
    
    var isVideoReady: Bool {
        guard let clip = selectedClip else { return false }
        return videoInfoCache[clip.url] != nil
    }

    private var playerItem: AVPlayerItem?
    private var observer: Any?
    private weak var observerPlayer: AVPlayer?
    private var playbackEndObserver: NSObjectProtocol?
    private var outputFolder: URL?
    private var outputFolderAccessing = false
    private var videoPreferredTransform: CGAffineTransform = .identity
    private var videoInfoCache: [URL: LoadedVideoInfo] = [:]
    private var isRestoringClipState = false
    private(set) var editingRevision = UUID()
    private let supportedVideoExtensions: Set<String> = ["mp4", "mov", "m4v", "avi", "mkv", "mts", "m2ts", "wmv"]
    
    init() {
        restoreOutputFolderFromBookmark()
    }

    deinit {
        MainActor.assumeIsolated {
            teardownObserver()
            for clip in clips where clip.isSecurityScoped {
                clip.url.stopAccessingSecurityScopedResource()
            }
            if outputFolderAccessing, let outputFolder {
                outputFolder.stopAccessingSecurityScopedResource()
            }
        }
    }
    
    var clipWidth: Double {
        clipEnd - clipStart
    }

    var clipRangeText: String {
        guard let duration = playerItem?.duration else { return "处理范围：全部" }
        let total = CMTimeGetSeconds(duration)
        guard total.isFinite, total > 0 else { return "处理范围：全部" }
        if clipStart <= 0.001 && clipEnd >= 0.999 {
            return "处理范围：全部"
        }
        let start = total * clipStart
        let end = total * clipEnd
        return "处理范围：\(Self.formatDisplayTime(start)) - \(Self.formatDisplayTime(end))"
    }

    var videoAspectRatio: CGFloat {
        let width = max(videoWidth, 1)
        let height = max(videoHeight, 1)
        return CGFloat(width / height)
    }

    func videoDisplayRect(in containerSize: CGSize) -> CGRect {
        guard containerSize.width > 0, containerSize.height > 0 else {
            return .zero
        }
        
        let aspect = videoAspectRatio
        let containerAspect = containerSize.width / max(containerSize.height, 1)
        
        let fittedRect: CGRect
        if containerAspect > aspect {
            let fittedHeight = containerSize.height
            let fittedWidth = fittedHeight * aspect
            fittedRect = CGRect(
                x: (containerSize.width - fittedWidth) / 2,
                y: 0,
                width: fittedWidth,
                height: fittedHeight
            )
        } else {
            let fittedWidth = containerSize.width
            let fittedHeight = fittedWidth / aspect
            fittedRect = CGRect(
                x: 0,
                y: (containerSize.height - fittedHeight) / 2,
                width: fittedWidth,
                height: fittedHeight
            )
        }

        let scaledWidth = fittedRect.width * zoom
        let scaledHeight = fittedRect.height * zoom
        return CGRect(
            x: (containerSize.width - scaledWidth) / 2,
            y: (containerSize.height - scaledHeight) / 2,
            width: scaledWidth,
            height: scaledHeight
        )
    }

    func normalizedVideoPoint(from location: CGPoint, in containerSize: CGSize) -> CGPoint {
        let rect = videoDisplayRect(in: containerSize)
        guard rect.width > 0, rect.height > 0 else { return .zero }
        let x = max(0, min(1, (location.x - rect.minX) / rect.width))
        let y = max(0, min(1, (location.y - rect.minY) / rect.height))
        return CGPoint(x: x, y: y)
    }

    private nonisolated static func displaySize(
        for naturalSize: CGSize,
        preferredTransform: CGAffineTransform
    ) -> CGSize {
        let transformed = CGRect(origin: .zero, size: naturalSize)
            .applying(preferredTransform)
            .standardized
        return CGSize(
            width: abs(transformed.width),
            height: abs(transformed.height)
        )
    }

    private func rawVideoPoint(fromDisplayPoint point: CGPoint) -> CGPoint {
        let rawSize = CGSize(width: rawVideoWidth, height: rawVideoHeight)
        guard rawSize.width > 0, rawSize.height > 0 else {
            return point
        }

        let transformedRect = CGRect(origin: .zero, size: rawSize)
            .applying(videoPreferredTransform)
            .standardized
        let displayPointBottom = CGPoint(
            x: transformedRect.minX + point.x,
            y: transformedRect.maxY - point.y
        )
        let rawBottom = displayPointBottom.applying(videoPreferredTransform.inverted())
        let rawTop = CGPoint(
            x: rawBottom.x,
            y: rawSize.height - rawBottom.y
        )

        return CGPoint(
            x: min(max(rawTop.x, 0), rawSize.width),
            y: min(max(rawTop.y, 0), rawSize.height)
        )
    }

    private func rawTrackingBoxSize(
        fromDisplayCenter center: CGPoint,
        displayBoxSize: CGSize
    ) -> CGSize {
        let halfWidth = displayBoxSize.width / 2
        let halfHeight = displayBoxSize.height / 2
        let rawCorners = [
            rawVideoPoint(fromDisplayPoint: CGPoint(x: center.x - halfWidth, y: center.y - halfHeight)),
            rawVideoPoint(fromDisplayPoint: CGPoint(x: center.x + halfWidth, y: center.y - halfHeight)),
            rawVideoPoint(fromDisplayPoint: CGPoint(x: center.x - halfWidth, y: center.y + halfHeight)),
            rawVideoPoint(fromDisplayPoint: CGPoint(x: center.x + halfWidth, y: center.y + halfHeight))
        ]
        let minX = rawCorners.map(\.x).min() ?? 0
        let maxX = rawCorners.map(\.x).max() ?? 0
        let minY = rawCorners.map(\.y).min() ?? 0
        let maxY = rawCorners.map(\.y).max() ?? 0
        return CGSize(
            width: max(24, abs(maxX - minX)),
            height: max(24, abs(maxY - minY))
        )
    }
    
    // MARK: - 导入
    func importVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = supportedVideoContentTypes()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            let response = panel.runModal()
            if response == .OK {
                self.importFiles(panel.urls)
            }
        }
    }

    func importProviders(_ providers: [NSItemProvider]) -> Bool {
        guard !providers.isEmpty else { return false }
        var importedURLs: [URL] = []
        let group = DispatchGroup()
        let resultQueue = DispatchQueue(label: "stableflow.importProviders")

        for provider in providers where provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            group.enter()
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                defer { group.leave() }

                if let data = item as? Data,
                   let url = NSURL(absoluteURLWithDataRepresentation: data, relativeTo: nil) as URL?,
                   isSupportedVideoURL(url) {
                    resultQueue.sync { importedURLs.append(url) }
                } else if let url = item as? URL, isSupportedVideoURL(url) {
                    resultQueue.sync { importedURLs.append(url) }
                } else if let str = item as? String,
                          let url = URL(string: str),
                          isSupportedVideoURL(url) {
                    resultQueue.sync { importedURLs.append(url) }
                }
            }
        }

        group.notify(queue: .main) { [weak self] in
            guard let self else { return }
            let uniqueURLs = Array(Set(importedURLs.map { $0.standardizedFileURL })).sorted { $0.path < $1.path }
            guard !uniqueURLs.isEmpty else { return }
            self.importFiles(uniqueURLs)
        }

        return true
    }
    
    private func importFiles(_ urls: [URL]) {
        let supportedURLs = urls
            .map { $0.standardizedFileURL }
            .filter(isSupportedVideoURL)
        guard !supportedURLs.isEmpty else { return }

        var existingPaths = Set(clips.map { $0.url.standardizedFileURL.path })
        let newURLs = supportedURLs.filter { existingPaths.insert($0.path).inserted }
        guard !newURLs.isEmpty else {
            if let first = supportedURLs.first,
               let index = clips.firstIndex(where: { $0.url.standardizedFileURL.path == first.path }) {
                selectClip(index)
            }
            return
        }

        for url in newURLs {
            let accessing = url.startAccessingSecurityScopedResource()
            let name = url.deletingPathExtension().lastPathComponent

            var clip = VideoClip(url: url, name: name, duration: "00:00:00")
            clip.isSecurityScoped = accessing
            clip.resolution = "加载中..."

            clips.append(clip)
            loadVideoInfo(for: clips.count - 1, url: url)
        }
        selectClip(clips.count - 1)
    }

    func importExternalFiles(_ urls: [URL]) {
        let uniqueURLs = Array(Set(urls.map { $0.standardizedFileURL })).sorted { $0.path < $1.path }
        guard !uniqueURLs.isEmpty else { return }
        importFiles(uniqueURLs)
    }

    private func loadVideoInfo(for index: Int, url: URL) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let asset = AVURLAsset(url: url)
            let info = Self.loadVideoInfoSync(for: asset)
            let durStr = Self.formatTimeStatic(CMTimeGetSeconds(info?.duration ?? .zero))
            
            var capturedResolution = ""
            var capturedFrameRate = ""
            var capturedFileSize = ""
            var videoWidth: Double = 1920
            var videoHeight: Double = 1080
            var rawVideoWidth: Double = 1920
            var rawVideoHeight: Double = 1080
            var preferredTransform: CGAffineTransform = .identity
            
            if let info {
                capturedResolution = "\(Int(info.displaySize.width))x\(Int(info.displaySize.height))"
                capturedFrameRate = "\(Int(info.nominalFrameRate)) fps"
                videoWidth = info.displaySize.width
                videoHeight = info.displaySize.height
                rawVideoWidth = info.naturalSize.width
                rawVideoHeight = info.naturalSize.height
                preferredTransform = info.preferredTransform
            }
            
            if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
               let size = attrs[.size] as? Int64 {
                capturedFileSize = ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
            }
            
            DispatchQueue.main.async {
                guard index < self.clips.count else { return }
                if let info {
                    self.videoInfoCache[url] = info
                }
                self.clips[index].duration = durStr
                self.clips[index].resolution = capturedResolution
                self.clips[index].frameRate = capturedFrameRate
                self.clips[index].fileSize = capturedFileSize
                
                // 更新当前视频尺寸
                if index == self.selectedIndex {
                    self.videoWidth = videoWidth
                    self.videoHeight = videoHeight
                    self.rawVideoWidth = rawVideoWidth
                    self.rawVideoHeight = rawVideoHeight
                    self.videoPreferredTransform = preferredTransform
                    self.durationString = durStr
                }
            }
        }
    }
    
    private nonisolated static func formatTimeStatic(_ seconds: Double) -> String {
        guard seconds.isFinite else { return "00:00:00" }
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        return String(format: "%02d:%02d:%02d", h, m, s)
    }

    private nonisolated static func loadVideoInfoSync(for asset: AVAsset) -> LoadedVideoInfo? {
        let semaphore = DispatchSemaphore(value: 0)
        var result: LoadedVideoInfo?

        Task {
            defer { semaphore.signal() }
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else { return }
                let duration = try await asset.load(.duration)
                let naturalSize = try await track.load(.naturalSize)
                let preferredTransform = try await track.load(.preferredTransform)
                let nominalFrameRate = try await track.load(.nominalFrameRate)
                result = LoadedVideoInfo(
                    duration: duration,
                    naturalSize: naturalSize,
                    preferredTransform: preferredTransform,
                    displaySize: Self.displaySize(for: naturalSize, preferredTransform: preferredTransform),
                    nominalFrameRate: nominalFrameRate
                )
            } catch {
                result = nil
            }
        }

        semaphore.wait()
        return result
    }
    
    private func codecToString(_ codec: FourCharCode) -> String {
        let chars = [
            Character(UnicodeScalar((codec >> 24) & 0xFF)!),
            Character(UnicodeScalar((codec >> 16) & 0xFF)!),
            Character(UnicodeScalar((codec >> 8) & 0xFF)!),
            Character(UnicodeScalar(codec & 0xFF)!)
        ]
        let str = String(chars)
        switch str {
        case "avc1": return "H.264"
        case "hvc1", "hev1": return "H.265/HEVC"
        case "av01": return "AV1"
        case "ap4h": return "ProRes 4444"
        case "ap4x": return "ProRes 4444 XQ"
        case "apch": return "ProRes 422 HQ"
        case "apcn": return "ProRes 422"
        case "apcs": return "ProRes 422 LT"
        case "apco": return "ProRes 422 Proxy"
        default: return str
        }
    }
    
    private func formatFileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    private func persistCurrentClipEditingState() {
        guard !isRestoringClipState,
              selectedIndex >= 0,
              selectedIndex < clips.count else { return }
        editingRevision = UUID()
        stabilizationAnalysis = nil
        clips[selectedIndex].trackerPoints = trackerPoints
        clips[selectedIndex].clipStart = clipStart
        clips[selectedIndex].clipEnd = clipEnd
    }
    
    func selectClip(_ idx: Int) {
        guard clips.indices.contains(idx) else { return }
        persistCurrentClipEditingState()
        selectedIndex = idx
        loadClip(idx)
    }
    
    private func loadClip(_ idx: Int) {
        guard idx >= 0 && idx < clips.count else { return }
        
        if !clips[idx].isSecurityScoped {
            clips[idx].isSecurityScoped = clips[idx].url.startAccessingSecurityScopedResource()
        }
        let clip = clips[idx]
        // Metadata is loaded off the main actor by loadVideoInfo. Never block
        // selection/import waiting for a second copy of the same metadata load.
        let info = videoInfoCache[clip.url]
        if let info {
            videoInfoCache[clip.url] = info
            videoWidth = info.displaySize.width
            videoHeight = info.displaySize.height
            rawVideoWidth = info.naturalSize.width
            rawVideoHeight = info.naturalSize.height
            videoPreferredTransform = info.preferredTransform
        } else {
            videoWidth = 1920
            videoHeight = 1080
            rawVideoWidth = 1920
            rawVideoHeight = 1080
            videoPreferredTransform = .identity
        }
        
        // 简单同步创建播放器
        let asset = AVURLAsset(url: clip.url)
        replacePlayer(with: AVPlayerItem(asset: asset))
        
        // 同步获取时长
        let dur = info?.duration ?? .zero
        durationString = CMTimeGetSeconds(dur).isFinite ? formatTime(CMTimeGetSeconds(dur)) : "00:00:00"
        
        progress = 0
        isRestoringClipState = true
        clipStart = clip.clipStart
        clipEnd = clip.clipEnd
        trackerPoints = clip.trackerPoints
        selectedTrackerIndex = trackerPoints.isEmpty ? nil : 0
        currentTimeString = "00:00:00"
        isRestoringClipState = false
        // 恢复该片段上次设置的跟踪框
        stabilizationAnalysis = nil
        
    }
    
    private func replacePlayer(with item: AVPlayerItem) {
        teardownObserver()
        playerItem = item
        let newPlayer = AVPlayer(playerItem: item)
        player = newPlayer
        isPlaying = false
        setupObserver()
    }

    private func teardownObserver() {
        if let obs = observer, let owner = observerPlayer {
            owner.removeTimeObserver(obs)
        }
        if let playbackEndObserver {
            NotificationCenter.default.removeObserver(playbackEndObserver)
        }
        observer = nil
        observerPlayer = nil
        playbackEndObserver = nil
    }

    private func setupObserver() {
        teardownObserver()
        guard let player else { return }

        observerPlayer = player
        observer = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.1, preferredTimescale: 600),
            queue: .main
        ) { [weak self, weak player] time in
            Task { @MainActor [weak self, weak player] in
                guard let self, let player, self.player === player else { return }
                let cur = CMTimeGetSeconds(time)
                let tot = CMTimeGetSeconds(self.playerItem?.duration ?? .zero)
                if tot.isFinite && tot > 0 {
                    self.progress = cur / tot
                    self.currentTimeString = Self.formatDisplayTime(cur)
                }
            }
        }

        // 循环播放：播放结束后自动从头开始
        playbackEndObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: player.currentItem,
            queue: .main
        ) { [weak self, weak player] _ in
            Task { @MainActor [weak self, weak player] in
                guard let self, let player, self.player === player, self.isPlaying else { return }
                self.player?.seek(to: .zero)
                self.player?.play()
            }
        }
    }
    
    // MARK: - 播放控制
    func togglePlay() {
        if isPlaying { player?.pause() }
        else { player?.play() }
        isPlaying.toggle()
    }
    
    func seekStart() {
        player?.seek(to: .zero)
        progress = 0
        currentTimeString = "00:00:00"
    }
    
    func seekEnd() {
        guard let dur = playerItem?.duration else { return }
        player?.seek(to: dur)
    }
    
    func seek(to p: Double) {
        guard let dur = playerItem?.duration else { return }
        let time = CMTime(seconds: CMTimeGetSeconds(dur) * p, preferredTimescale: 600)
        player?.seek(to: time)
    }

    func seekRelative(by seconds: Double) {
        guard let player, let dur = playerItem?.duration else { return }
        let durationSeconds = CMTimeGetSeconds(dur)
        guard durationSeconds.isFinite, durationSeconds > 0 else { return }

        let current = CMTimeGetSeconds(player.currentTime())
        let target = max(0, min(durationSeconds, current + seconds))
        let time = CMTime(seconds: target, preferredTimescale: 600)
        player.seek(to: time)
        progress = target / durationSeconds
        currentTimeString = Self.formatDisplayTime(target)
    }

    func setClipStartToCurrentTime() {
        guard let player, let dur = playerItem?.duration else { return }
        let durationSeconds = CMTimeGetSeconds(dur)
        guard durationSeconds.isFinite, durationSeconds > 0 else { return }

        let current = CMTimeGetSeconds(player.currentTime()) / durationSeconds
        clipStart = max(0, min(clipEnd - 0.05, current))
    }

    func setClipEndToCurrentTime() {
        guard let player, let dur = playerItem?.duration else { return }
        let durationSeconds = CMTimeGetSeconds(dur)
        guard durationSeconds.isFinite, durationSeconds > 0 else { return }

        let current = CMTimeGetSeconds(player.currentTime()) / durationSeconds
        clipEnd = min(1, max(clipStart + 0.05, current))
    }

    func resetClipRange() {
        clipStart = 0
        clipEnd = 1
    }
    
    // MARK: - 缩放
    func adjustZoom(by delta: CGFloat) {
        let factor = 1 + Double(delta) * 0.003
        applyZoomFactor(factor)
    }

    func adjustZoomByMagnification(_ magnification: CGFloat) {
        let factor = 1 + Double(magnification)
        applyZoomFactor(factor)
    }

    private func applyZoomFactor(_ factor: Double) {
        guard factor.isFinite, factor > 0 else { return }
        zoom = max(0.5, min(4.0, zoom * factor))
    }
    
    // MARK: - 跟踪点
    var trackingSelectionTime: Double {
        let seconds = player.map { CMTimeGetSeconds($0.currentTime()) } ?? 0
        return seconds.isFinite ? seconds : 0
    }

    func pauseForTrackingSelection() {
        player?.pause()
        isPlaying = false
    }

    nonisolated static func isTrackingReferenceAligned(_ referenceTime: Double, clipStartTime: Double, fps: Double) -> Bool {
        guard referenceTime.isFinite, clipStartTime.isFinite, fps.isFinite, fps > 0 else { return false }
        return abs(referenceTime - clipStartTime) <= 1 / fps + 0.001
    }

    func addTrackerPoint() {
        guard videoWidth > 0, videoHeight > 0 else { return }
        let newPoint = TrackerPoint(
            position: CGPoint(x: videoWidth / 2, y: videoHeight / 2),
            boxSize: CGSize(width: 80, height: 80),
            referenceTime: trackingSelectionTime
        )
        trackerPoints = [newPoint]
        selectedTrackerIndex = 0
    }
    
    func setTrackerPoint(_ position: CGPoint, in size: CGSize) {
        let normalized = normalizedVideoPoint(from: position, in: size)
        let videoX = normalized.x * videoWidth
        let videoY = normalized.y * videoHeight
        
        if trackerPoints.isEmpty {
            trackerPoints.append(TrackerPoint(
                position: CGPoint(x: videoX, y: videoY),
                boxSize: CGSize(width: 80, height: 80),
                referenceTime: trackingSelectionTime
            ))
        } else {
            trackerPoints[0].position = CGPoint(x: videoX, y: videoY)
            trackerPoints[0].referenceTime = trackingSelectionTime
        }
        selectedTrackerIndex = 0
        stabilizationAnalysis = nil
    }
    
    func moveTrackerPoint(_ idx: Int, to position: CGPoint, in size: CGSize) {
        guard trackerPoints.indices.contains(idx) else { return }
        
        let normalized = normalizedVideoPoint(from: position, in: size)
        let videoX = normalized.x * videoWidth
        let videoY = normalized.y * videoHeight
        
        trackerPoints[idx].position = CGPoint(x: videoX, y: videoY)
        trackerPoints[idx].referenceTime = trackingSelectionTime
        stabilizationAnalysis = nil
    }
    
    func removeTrackerPoint(_ idx: Int) {
        guard trackerPoints.indices.contains(idx) else { return }
        trackerPoints.remove(at: idx)
        selectedTrackerIndex = trackerPoints.isEmpty ? nil : 0
        stabilizationAnalysis = nil
    }
    
    // MARK: - 处理
    func processVideo() {
        guard !isProcessing, let clip = selectedClip else { return }
        persistCurrentClipEditingState()
        guard !trackerPoints.isEmpty else {
            processStatus = "请先在预览区拖拽框选要锁定的特征"
            return
        }
        guard let info = videoInfoCache[clip.url],
              let tracker = trackerPoints.first else {
            processStatus = "视频信息尚未就绪"
            return
        }
        let durationSeconds = CMTimeGetSeconds(info.duration)
        let fps = Double(info.nominalFrameRate)
        guard durationSeconds.isFinite, durationSeconds > 0, fps.isFinite, fps > 0 else {
            processStatus = "错误：无效视频时长或帧率"
            return
        }
        let startSeconds = durationSeconds * clipStart
        guard Self.isTrackingReferenceAligned(
            tracker.referenceTime,
            clipStartTime: startSeconds,
            fps: fps
        ) else {
            pauseForTrackingSelection()
            player?.seek(
                to: CMTime(seconds: startSeconds, preferredTimescale: 600),
                toleranceBefore: .zero,
                toleranceAfter: .zero
            )
            processStatus = "已定位到片段起点，请在此帧重新框选特征"
            return
        }
        guard ensureOutputFolder() else {
            processStatus = "未选择或无法写入输出目录"
            return
        }

        isProcessing = true
        processProgress = 0
        processStatus = "分析运动..."

        let outputURL = processedOutputURL(for: clip)
        let config = makeProcessingConfig()
        let selectedClipID = clip.id
        let processingRevision = editingRevision
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let (success, analysis, warning) = Self.runProcessing(
                input: clip.url,
                output: outputURL,
                config: config
            ) { progress, status in
                DispatchQueue.main.async {
                    self.processProgress = progress
                    self.processStatus = status
                }
            }
            if config.debugMode || !success {
                Self.writeProcessingDebugReport(
                    input: clip.url,
                    output: outputURL,
                    config: config,
                    success: success,
                    analysis: analysis
                )
            }
            
            DispatchQueue.main.async {
                self.finishProcessing(
                    clipID: selectedClipID,
                    outputURL: outputURL,
                    success: success,
                    analysis: analysis,
                    expectedEditingRevision: processingRevision,
                    warning: warning
                )
            }
        }
    }
    
    // Keep the editor on its source timeline. A cropped/trimmed export has different
    // coordinates and duration, and another clip may have been selected meanwhile.
    func finishProcessing(
        clipID: UUID,
        outputURL: URL,
        success: Bool,
        analysis: StabilizationAnalysis?,
        expectedEditingRevision: UUID? = nil,
        warning: String? = nil
    ) {
        isProcessing = false
        processStatus = success ? (warning.map { "完成（\($0)）" } ?? "完成") : "处理失败"
        if selectedClip?.id == clipID,
           expectedEditingRevision == nil || expectedEditingRevision == editingRevision {
            stabilizationAnalysis = analysis
        }
        if success, let idx = clips.firstIndex(where: { $0.id == clipID }) {
            clips[idx].isProcessed = true
            clips[idx].processedURL = outputURL
        }
    }

    @discardableResult
    private func ensureOutputFolder() -> Bool {
        if let outputFolder {
            if !outputFolderAccessing {
                outputFolderAccessing = outputFolder.startAccessingSecurityScopedResource()
            }
            return validateWritableOutputFolder(outputFolder)
        }

        return chooseOutputFolder()
    }

    private func processedOutputURL(for clip: VideoClip) -> URL {
        Self.availableOutputURL(for: clip.name, in: outputFolder!, format: exportFormat)
    }

    nonisolated static func availableOutputURL(for name: String, in folder: URL, format: ExportFormat) -> URL {
        let baseName = "\(name)_stable"
        var candidate = folder.appendingPathComponent("\(baseName).\(format.fileExtension)")
        var suffix = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = folder.appendingPathComponent("\(baseName)_\(suffix).\(format.fileExtension)")
            suffix += 1
        }
        return candidate
    }

    @discardableResult
    func chooseOutputFolder() -> Bool {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "选择稳定视频导出目录"
        panel.prompt = "选择目录"

        let response = panel.runModal()
        guard response == .OK, let url = panel.url else { return false }
        setOutputFolder(url)
        return validateWritableOutputFolder(url)
    }

    private func setOutputFolder(_ url: URL) {
        if outputFolderAccessing, let outputFolder {
            outputFolder.stopAccessingSecurityScopedResource()
        }
        outputFolder = url
        outputFolderAccessing = url.startAccessingSecurityScopedResource()
        do {
            let bookmark = try url.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            UserDefaults.standard.set(bookmark, forKey: outputFolderBookmarkDefaultsKey)
        } catch {
            UserDefaults.standard.removeObject(forKey: outputFolderBookmarkDefaultsKey)
        }
        _ = validateWritableOutputFolder(url)
    }

    private func restoreOutputFolderFromBookmark() {
        guard let bookmark = UserDefaults.standard.data(forKey: outputFolderBookmarkDefaultsKey) else {
            return
        }

        var isStale = false
        guard let url = try? URL(
            resolvingBookmarkData: bookmark,
            options: [.withSecurityScope],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        ) else {
            UserDefaults.standard.removeObject(forKey: outputFolderBookmarkDefaultsKey)
            return
        }

        setOutputFolder(url)
        if isStale {
            UserDefaults.standard.removeObject(forKey: outputFolderBookmarkDefaultsKey)
            do {
                let refreshed = try url.bookmarkData(
                    options: [.withSecurityScope],
                    includingResourceValuesForKeys: nil,
                    relativeTo: nil
                )
                UserDefaults.standard.set(refreshed, forKey: outputFolderBookmarkDefaultsKey)
            } catch {
                UserDefaults.standard.removeObject(forKey: outputFolderBookmarkDefaultsKey)
            }
        }
    }

    private func validateWritableOutputFolder(_ url: URL) -> Bool {
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let probeURL = url.appendingPathComponent(".stableflow_write_test_\(UUID().uuidString)")
            try Data().write(to: probeURL, options: .atomic)
            try? FileManager.default.removeItem(at: probeURL)
            return true
        } catch {
            return false
        }
    }

    private func makeProcessingConfig() -> ProcessingConfig {
        let displayPoint = trackerPoints.first?.position
        let rawPoint = displayPoint.map(rawVideoPoint(fromDisplayPoint:))
        let rawBoxSize: CGSize
        if let displayPoint, let displayBoxSize = trackerPoints.first?.boxSize {
            rawBoxSize = rawTrackingBoxSize(
                fromDisplayCenter: displayPoint,
                displayBoxSize: displayBoxSize
            )
        } else {
            rawBoxSize = CGSize(width: 80, height: 80)
        }

        return ProcessingConfig(
            trackingPoint: rawPoint,
            trackingBoxSize: rawBoxSize,
            exportFormat: exportFormat.stabilizerValue,
            processingQuality: quality.rawValue,
            smoothness: smoothness,
            stabilizeStrength: stabilizeStrength,
            rollingShutterCompensation: rollingShutter,
            clipStart: clipStart,
            clipEnd: clipEnd,
            debugMode: showDebug
        )
    }

    // 已停用的旧 CPU 简化路径。保留代码只作为回滚参考，
    // 当前产品主链已经统一到 ImprovedStabilizer。
    private func processClipCPU(input: URL, output: URL) -> Bool {
        processStatus = "旧 CPU 路径已停用"
        return false
    }
    
    func revealProcessedVideo() {
        if let url = selectedClip?.processedURL, FileManager.default.fileExists(atPath: url.path) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else if let folder = outputFolder {
            NSWorkspace.shared.open(folder)
        }
    }
    
    private nonisolated static func formatDisplayTime(_ seconds: Double) -> String {
        guard seconds.isFinite else { return "00:00:00" }
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        
        if h > 0 {
            return String(format: "%02d:%02d:%02d", h, m, s)
        } else {
            return String(format: "%02d:%02d", m, s)
        }
    }

    private func formatTime(_ seconds: Double) -> String {
        Self.formatDisplayTime(seconds)
    }

    private nonisolated static func runProcessing(
        input: URL,
        output: URL,
        config: ProcessingConfig,
        progressHandler: @escaping (Double, String) -> Void
    ) -> (success: Bool, analysis: StabilizationAnalysis?, warning: String?) {
        guard let stabilizer = ImprovedStabilizer() as ImprovedStabilizer? else {
            progressHandler(1.0, "错误：稳定器未初始化")
            return (false, nil, nil)
        }
        
        stabilizer.debugMode = config.debugMode
        let success = stabilizer.stabilizeVideo(
            inputURL: input,
            outputURL: output,
            trackingPoint: config.trackingPoint,
            trackingBoxSize: config.trackingBoxSize,
            exportFormat: config.exportFormat,
            processingQuality: config.processingQuality,
            smoothFactor: config.smoothness,
            stabilizeStrength: config.stabilizeStrength,
            rollingShutterCompensation: config.rollingShutterCompensation,
            clipStart: config.clipStart,
            clipEnd: config.clipEnd,
            progressHandler: progressHandler
        )
        
        return (success, stabilizer.lastAnalysis, stabilizer.lastWarning)
    }

    private nonisolated static func writeProcessingDebugReport(
        input: URL,
        output: URL,
        config: ProcessingConfig,
        success: Bool,
        analysis: StabilizationAnalysis?
    ) {
        let reportURL = output.deletingPathExtension().appendingPathExtension("debug.txt")
        var lines: [String] = []
        lines.append("StableFlow Pro Processing Debug Report")
        lines.append("input=\(input.path)")
        lines.append("output=\(output.path)")
        lines.append("success=\(success)")
        if let trackingPoint = config.trackingPoint {
            lines.append(String(format: "trackingPoint=(%.3f, %.3f)", trackingPoint.x, trackingPoint.y))
        } else {
            lines.append("trackingPoint=nil")
        }
        lines.append(String(format: "trackingBoxSize=(%.3f, %.3f)", config.trackingBoxSize.width, config.trackingBoxSize.height))
        lines.append("exportFormat=\(config.exportFormat)")
        lines.append("processingQuality=\(config.processingQuality)")
        lines.append(String(format: "smoothness=%.3f", config.smoothness))
        lines.append(String(format: "stabilizeStrength=%.3f", config.stabilizeStrength))
        lines.append("rollingShutterCompensation=\(config.rollingShutterCompensation)")
        lines.append(String(format: "clipStart=%.6f", config.clipStart))
        lines.append(String(format: "clipEnd=%.6f", config.clipEnd))
        lines.append("debugMode=\(config.debugMode)")
        if let analysis {
            lines.append(String(format: "meanCenterError=%.6f", analysis.meanCenterError))
            lines.append(String(format: "maxCenterError=%.6f", analysis.maxCenterError))
            lines.append(String(format: "cropRatio=%.6f", analysis.cropRatio))
            lines.append("cropLimited=\(analysis.cropLimited)")
            lines.append(String(format: "meanTransformDelta=%.6f", analysis.meanTransformDelta))
            lines.append(String(format: "maxTransformDelta=%.6f", analysis.maxTransformDelta))
            let transformPreview = analysis.transforms.prefix(8).map {
                String(format: "(%.2f, %.2f)", $0.x, $0.y)
            }.joined(separator: ", ")
            lines.append("transformPreview=\(transformPreview)")
        } else {
            lines.append("analysis=nil")
        }
        let text = lines.joined(separator: "\n") + "\n"
        try? text.write(to: reportURL, atomically: true, encoding: .utf8)
    }
}
