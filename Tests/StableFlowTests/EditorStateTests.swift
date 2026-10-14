import XCTest
import AVFoundation
@testable import StableFlow

final class EditorStateTests: XCTestCase {
    func testOutputNamesNeverReuseExistingExports() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let first = ProVideoVM.availableOutputURL(for: "clip", in: folder, format: .mp4)
        XCTAssertEqual(first.lastPathComponent, "clip_stable.mp4")
        try Data("first export".utf8).write(to: first)
        let second = ProVideoVM.availableOutputURL(for: "clip", in: folder, format: .mp4)
        XCTAssertEqual(second.lastPathComponent, "clip_stable_2.mp4")
        try Data().write(to: second)
        XCTAssertEqual(ProVideoVM.availableOutputURL(for: "clip", in: folder, format: .mp4).lastPathComponent, "clip_stable_3.mp4")
        XCTAssertEqual(try String(contentsOf: first), "first export")
        XCTAssertEqual(ProVideoVM.availableOutputURL(for: "clip", in: folder, format: .prores).pathExtension, "mov")
    }

    func testTrackingReferenceMustMatchClipStart() {
        XCTAssertTrue(ProVideoVM.isTrackingReferenceAligned(2, clipStartTime: 2, fps: 30))
        XCTAssertTrue(ProVideoVM.isTrackingReferenceAligned(2.02, clipStartTime: 2, fps: 30))
        XCTAssertFalse(ProVideoVM.isTrackingReferenceAligned(5, clipStartTime: 2, fps: 30))
        XCTAssertFalse(ProVideoVM.isTrackingReferenceAligned(2, clipStartTime: 5, fps: 30))
        XCTAssertFalse(ProVideoVM.isTrackingReferenceAligned(.nan, clipStartTime: 0, fps: 30))
        XCTAssertFalse(ProVideoVM.isTrackingReferenceAligned(0, clipStartTime: 0, fps: 0))
    }

    func testCompletionKeepsCurrentSourcePlayerAndSelection() async {
        await MainActor.run {
            let vm = ProVideoVM()
            let first = VideoClip(url: URL(fileURLWithPath: "/tmp/first.mov"), name: "first", duration: "00:10")
            let second = VideoClip(url: URL(fileURLWithPath: "/tmp/second.mov"), name: "second", duration: "00:20")
            vm.clips = [first, second]
            vm.selectedIndex = 1
            let sourcePlayer = AVPlayer()
            vm.player = sourcePlayer
            vm.clipStart = 0.2
            vm.clipEnd = 0.8
            vm.isProcessing = true
            let output = URL(fileURLWithPath: "/tmp/first_stable.mov")
            vm.finishProcessing(clipID: first.id, outputURL: output, success: true, analysis: nil)
            XCTAssertFalse(vm.isProcessing)
            XCTAssertEqual(vm.selectedIndex, 1)
            XCTAssertTrue(vm.player === sourcePlayer)
            XCTAssertEqual(vm.clipStart, 0.2)
            XCTAssertEqual(vm.clipEnd, 0.8)
            XCTAssertTrue(vm.clips[0].isProcessed)
            XCTAssertEqual(vm.clips[0].processedURL, output)
            XCTAssertFalse(vm.clips[1].isProcessed)
            // The same clip must also retain source coordinates after a trimmed export.
            vm.selectedIndex = 0
            vm.finishProcessing(clipID: first.id, outputURL: output, success: true, analysis: nil)
            XCTAssertTrue(vm.player === sourcePlayer)
        }
    }

    func testFailedProcessingDoesNotMarkClipProcessed() async {
        await MainActor.run {
            let vm = ProVideoVM()
            let clip = VideoClip(url: URL(fileURLWithPath: "/tmp/input.mov"), name: "input", duration: "00:10")
            vm.clips = [clip]
            vm.selectedIndex = 0
            vm.finishProcessing(clipID: clip.id, outputURL: URL(fileURLWithPath: "/tmp/output.mov"), success: false, analysis: nil)
            XCTAssertFalse(vm.clips[0].isProcessed)
            XCTAssertNil(vm.clips[0].processedURL)
        }
    }

    func testCompletionDoesNotRestoreAnalysisAfterEditing() async {
        await MainActor.run {
            let vm = ProVideoVM()
            let clip = VideoClip(url: URL(fileURLWithPath: "/tmp/input.mov"), name: "input", duration: "00:10")
            vm.clips = [clip]
            vm.selectedIndex = 0
            let revision = vm.editingRevision
            let analysis = StabilizationAnalysis(
                frameSize: CGSize(width: 100, height: 100), rawTrajectory: [],
                smoothedTrajectory: [], renderedTrajectory: [], transforms: [],
                isCentering: true, cropRatio: 0, cropLimited: false,
                meanCenterError: 0, maxCenterError: 0,
                meanTransformDelta: 0, maxTransformDelta: 0)
            vm.clipStart = 0.2
            vm.finishProcessing(
                clipID: clip.id, outputURL: URL(fileURLWithPath: "/tmp/output.mov"),
                success: true, analysis: analysis, expectedEditingRevision: revision,
                warning: "audio unavailable")
            XCTAssertNil(vm.stabilizationAnalysis)
            XCTAssertTrue(vm.processStatus.contains("audio unavailable"))
            XCTAssertTrue(vm.clips[0].isProcessed)
        }
    }

    func testInvalidTrackerIndicesAndRepeatedProcessingAreIgnored() async {
        await MainActor.run {
            let vm = ProVideoVM()
            vm.trackerPoints = [TrackerPoint(position: CGPoint(x: 10, y: 20))]
            vm.selectedTrackerIndex = 0
            vm.moveTrackerPoint(-1, to: .zero, in: CGSize(width: 100, height: 100))
            vm.removeTrackerPoint(-1)
            XCTAssertEqual(vm.trackerPoints.count, 1)
            vm.removeTrackerPoint(0)
            XCTAssertNil(vm.selectedTrackerIndex)
            vm.isProcessing = true
            vm.processStatus = "existing job"
            vm.processVideo()
            XCTAssertEqual(vm.processStatus, "existing job")
        }
    }
}
